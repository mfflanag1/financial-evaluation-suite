"""Schedule and host-journal bindings are required before financial aggregation."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from test_runtime_release_accounting import trace
from financial_portfolio_operations.telemetry import RequestJournal
from financial_portfolio_runtime.accounting import report_path
from financial_portfolio_runtime.telemetry import RuntimeJournal, digest, prompt_binding


def sha(value):
    return hashlib.sha256(value).hexdigest()


class RuntimeBoundAccounting(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.journal = self.root / "journal"
        self.journal.mkdir()
        self.record = trace()
        data = {"id": "task-one", "public_files": {"public.csv": "x\n1\n"},
                "prompt": "Task", "system_prompt": "Rules", "image": "sha256:" + "a" * 64}
        self.record["task"] = {"data": data}
        self.record["agent"] = {"runtime": {"image": data["image"]}}
        messages = [{"role": "system", "content": "Runtime rules"}, {"role": "user", "content": "Task"}]
        self.record["nodes"] = [{"message": m} for m in messages]
        self.record["calls"] = [{"model": "fixture-model", "sampling": {}}]
        runtime = RuntimeJournal(self.journal / "lifecycle.jsonl")
        provider = RequestJournal(self.journal / "provider")
        try:
            native = SimpleNamespace(id=self.record["id"], info=self.record["info"])
            runtime.bind(native, messages)
            events = [runtime.emit(native.id, "harness_started")]
            runtime.emit(native.id, "outbound_prompt_checked", request_index=1, valid=True,
                         messages_sha256=digest(messages))
            provider.register(trace_id=native.id, task_id=data["id"], model="fixture-model")
            request = provider.start(native.id, transport="stream")
            request.finish("success")
            events.extend([runtime.emit(native.id, "harness_terminal", outcome="returned", exit_code=0),
                           runtime.emit(native.id, "runtime_health", alive=True)])
            native.info["portfolio_runtime"]["events"] = events
            config = {"model": "fixture-model"}
            (self.journal / "bound.json").write_text(json.dumps(config))
            self.binding = {"schema_version": 1, "protocol_id": "portfolio-runtime-v1", "dry_run": False,
                            "expected_attempts": 1, "capture_id": runtime.run_id, "provider_run_id": provider.run_id,
                            "config_sha256": sha((self.journal / "bound.json").read_bytes()),
                            "model": "fixture-model", "sampling": {}, "docker_image_id": data["image"],
                            "schedule": [{"task_id": data["id"],
                                          "public_sha256": {name: sha(value.encode()) for name, value in data["public_files"].items()},
                                          "prompt_sha256": sha(data["prompt"].encode()),
                                          "system_prompt_sha256": sha(data["system_prompt"].encode()),
                                          "outbound_prompt": prompt_binding(messages)}]}
            self.save_binding()
        finally:
            runtime.close()
            provider.close()

    def save_binding(self):
        (self.journal / "run_binding.json").write_text(json.dumps(self.binding))

    def report(self, records=None, expected=1):
        path = self.root / "traces.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in (records or [self.record])) + "\n")
        return report_path(path, expected=expected, journal_dir=self.journal)

    def test_coherent_bound_attempt_is_usable(self):
        report = self.report()
        self.assertTrue(report["performance_usable"], report)
        self.assertEqual(report["mean_reward"], 0)

    def test_four_distinct_records_of_same_task_cannot_satisfy_four_task_schedule(self):
        self.binding["expected_attempts"] = 4
        template = self.binding["schedule"][0]
        self.binding["schedule"] = [{**template, "task_id": f"task-{i}"} for i in range(4)]
        self.save_binding()
        records = [{**deepcopy(self.record), "id": f"trace-{i}"} for i in range(4)]
        report = self.report(records, expected=4)
        self.assertIn("task_schedule_mismatch", report["integrity_issues"])
        self.assertFalse(report["performance_usable"])
        self.assertIsNone(report["mean_reward"])

    def test_native_public_input_change_blocks_aggregate(self):
        self.record["task"]["data"]["public_files"]["public.csv"] += "2\n"
        report = self.report()
        self.assertIn("task_input_binding_mismatch", report["entries"][0]["binding_issues"])
        self.assertIsNone(report["mean_reward"])

    def test_deleted_host_prompt_check_blocks_aggregate(self):
        path = self.journal / "lifecycle.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        path.write_text("\n".join(json.dumps(row) for row in rows if row["event"] != "outbound_prompt_checked") + "\n")
        report = self.report()
        self.assertIn("outbound_prompt_checks_invalid", report["entries"][0]["binding_issues"])
        self.assertIsNone(report["mean_reward"])

    def test_trace_lifecycle_cannot_diverge_from_host_journal(self):
        self.record["info"]["portfolio_runtime"]["events"].pop()
        self.assertIn("lifecycle_trace_mismatch", self.report()["entries"][0]["binding_issues"])

    def test_tool_summary_cannot_claim_a_result_absent_from_host_journal(self):
        info = self.record["info"]["portfolio_runtime"]
        occurrence = "digest:call"
        additions = [
            {"event": "tool_gate_started", "occurrence": occurrence, "tool_name": "bash"},
            {"event": "tool_gate_terminal", "occurrence": occurrence, "status": "allowed"},
        ]
        path = self.journal / "lifecycle.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for event in additions:
            event.update(trace_id=self.record["id"], run_id=self.binding["capture_id"])
            rows.append(event)
            info["events"].append(event)
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        info["tools"] = {occurrence: {"status": "result_observed", "tool_name": "bash", "tool_call_id": "call"}}
        report = self.report()
        self.assertIn("tool_state_lifecycle_mismatch", report["entries"][0]["binding_issues"])
        self.assertIsNone(report["mean_reward"])

    def test_wrong_provider_task_identity_blocks_aggregate(self):
        path = self.journal / "provider/requests.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            row["task_id"] = "another-task"
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        self.assertIn("provider_trace_correspondence_mismatch", self.report()["entries"][0]["binding_issues"])

    def test_changed_run_config_and_missing_binding_block_aggregate(self):
        (self.journal / "bound.json").write_text('{"model":"another-model"}')
        self.assertIn("invalid_run_binding", self.report()["integrity_issues"])
        (self.journal / "run_binding.json").unlink()
        self.assertIsNone(self.report()["mean_reward"])


if __name__ == "__main__":
    unittest.main()
