"""Offline native-hook regressions for the separately versioned staged workflow."""

import asyncio
import copy
from contextlib import asynccontextmanager
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
for package in ("financial_research", "financial_portfolio", "financial_portfolio_staged"):
    sys.path.insert(0, str(ROOT / "environments" / package / "src"))
try:
    import verifiers.v1 as vf
except ModuleNotFoundError as error:
    if error.name != "verifiers":
        raise
    vf = None

if vf is not None:
    from verifiers.v1 import graph
    from verifiers.v1.clients import ModelContext
    from verifiers.v1.errors import SandboxError
    from verifiers.v1.session import RolloutSession, hook_boundary
    from verifiers.v1.utils.decorators import discover_decorated
    from financial_research import prime_adapter as base
    from financial_portfolio import prime_adapter as frozen
    from financial_portfolio_staged import prime_adapter as adapter
    from financial_portfolio_staged.validation import TRIAL_FIELDS, inventory_for_public_files


def archive(path, payload=b"", *, kind=tarfile.REGTYPE, name=None, declared_size=None):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as bundle:
        entry = tarfile.TarInfo(name or path.lstrip("/"))
        entry.type = kind
        entry.linkname = "/tmp/foreign" if kind == tarfile.SYMTYPE else ""
        entry.size = len(payload) if declared_size is None else declared_size
        bundle.addfile(entry, io.BytesIO(payload) if kind == tarfile.REGTYPE else None)
    return buffer.getvalue()


class RecordingRuntime:
    type = "docker"
    config = SimpleNamespace(mounts={})

    def __init__(self, name="staged-fixture-runtime", payloads=None):
        self.name, self.files, self.payloads = name, {}, payloads or {}
        self.commands = []

    async def write(self, path, content):
        self.files[path] = content

    async def run(self, *args, **kwargs):
        self.commands.append((args, kwargs))
        raise AssertionError("Candidate commands must never execute in these host-side tests")


@unittest.skipUnless(vf is not None, "Install the pinned prime optional dependency")
class StagedAdapter(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        key = patch.dict(os.environ, {
            "FINANCIAL_RESEARCH_KEY": "staged-adapter-test-key-not-for-evaluation-0000"})
        key.start()
        self.addCleanup(key.stop)
        self.task = next(iter(adapter.FinancialPortfolioStagedTaskset(
            frozen.PortfolioTasksetConfig(num_tasks=1))))
        self.trace = self.new_trace(self.task)
        self.payloads = self.structural_payloads(self.task)
        self.runtime = RecordingRuntime(payloads=self.payloads)
        self.collect = AsyncMock(side_effect=self.collect_artifacts)
        collect = patch.object(adapter.vf, "collect", self.collect)
        collect.start()
        self.addCleanup(collect.stop)
        self.addCleanup(adapter._SOLVERS.clear)

    @staticmethod
    def new_trace(task):
        return vf.Trace(task=vf.TraceTask(type=type(task).__name__, data=task.data),
                        agent=vf.AgentInfo(config=vf.AgentConfig(runtime=vf.DockerConfig())),
                        state=base.ResearchState())

    @staticmethod
    def structural_payloads(task):
        contract = json.loads(task.data.public_files["contract.json"])
        trials = [{"id": candidate["id"], **{field: 0.0 for field in TRIAL_FIELDS}}
                  for candidate in contract["candidates"]]
        policy = {"audit": [{**row, "value": None} for row in contract["audit_requests"]],
                  "trials": trials, "decision": "abstain", "selected": None}
        return {
            "/workspace/research.py": b"raise RuntimeError('test fixture must never execute')\n",
            "/workspace/ingestion.json": json.dumps(inventory_for_public_files(task.data.public_files)).encode(),
            "/workspace/trials.json": json.dumps(trials).encode(),
            "/workspace/policy.json": json.dumps(policy).encode(),
        }

    @staticmethod
    async def collect_artifacts(runtime, artifacts, **kwargs):
        await asyncio.sleep(0)
        return {artifact.source: archive(artifact.source, runtime.payloads[artifact.source])
                if artifact.source in runtime.payloads else None for artifact in artifacts}

    def session(self, task=None, trace=None):
        task, trace = task or self.task, trace or self.trace
        stops = [(hook_boundary(fn, allow_trace=True), fn) for fn in task.hooks("stop")]
        hooks = [(hook_boundary(fn, allow_trace=False), fn)
                 for fn in discover_decorated(task, "intercept")]
        return RolloutSession(
            ctx=ModelContext(model="offline-fixture", client=vf.EvalClientConfig()), trace=trace,
            trace_stops=[fn for kind, fn in stops if kind is vf.Trace],
            request_stops=[fn for kind, fn in stops if kind is vf.Request],
            response_stops=[fn for kind, fn in stops if kind is vf.Response],
            request_interceptors=[fn for kind, fn in hooks if kind is vf.Request],
            response_interceptors=[fn for kind, fn in hooks if kind is vf.Response], gates_tools=True)

    @staticmethod
    def response(calls, trace):
        return vf.Response(id="offline-response", created=0, model="offline-fixture",
                           message=vf.AssistantMessage(content="", tool_calls=calls),
                           finish_reason="tool_calls" if calls else "stop", path_len=0)

    @staticmethod
    def call(command, call_id="fixture-call", *, arguments=None):
        return vf.ToolCall(id=call_id, name="bash",
                           arguments=json.dumps({"command": command}) if arguments is None else arguments)

    async def commit(self, session, calls):
        response = self.response(calls, session.trace)
        original = response.model_dump()
        rewritten, records, stopped = await session.rewrite_response(response)
        self.assertEqual(rewritten.model_dump(), original)
        self.assertEqual(records, [])
        prompt = session.trace.messages or [vf.UserMessage(content="offline fixture")]
        node = graph.prepare_turn(session.trace, prompt).commit(rewritten)
        if stopped:
            session.trace.stop(stopped)
        else:
            gated_stop = await session.gate_tool_calls(node)
            if gated_stop:
                session.trace.stop(gated_stop)
        return node

    async def checkpoint(self, stage, *, task=None, trace=None, session=None):
        task, trace = task or self.task, trace or self.trace
        session = session or self.session(task, trace)
        call = self.call(f"financial-checkpoint {stage}", f"checkpoint-{len(trace.nodes)}")
        await self.commit(session, [call])
        decision = await session.decide_tool(call.id, call.name, json.loads(call.arguments))
        self.assertEqual(decision["action"], "deny")
        message = vf.ToolMessage.model_validate(decision["message"])
        request = vf.Request(messages=[*trace.messages, message])
        rewritten, _, stopped = await session.rewrite_request(request)
        self.assertIsNone(stopped)
        graph.prepare_turn(trace, rewritten.messages).commit_prompt()
        return session, json.loads(message.content)

    def test_native_loader_public_record_and_frozen_data_identity(self):
        from verifiers.v1.utils.loaders import taskset_class

        self.assertIs(taskset_class("financial-portfolio-staged"), adapter.FinancialPortfolioStagedTaskset)
        original = next(iter(frozen.FinancialPortfolioTaskset(frozen.PortfolioTasksetConfig(num_tasks=1))))
        excluded = {"system_prompt", "artifacts", "artifact_max_bytes"}
        self.assertEqual(self.task.data.model_dump(exclude=excluded), original.data.model_dump(exclude=excluded))
        self.assertEqual(self.task.config, original.config)
        self.assertEqual(set(self.task.data.model_dump()) - set(vf.TaskData.model_fields), {"public_files"})
        data = self.task.data.model_dump()
        self.assertFalse({"episode", "answer", "reward", "seed", "family", "hidden"} & set(data))
        self.assertNotIn("staged-adapter-test-key", self.task.data.model_dump_json())
        self.assertEqual([artifact.source for artifact in self.task.data.artifacts], list(adapter.PATHS))
        self.assertEqual(self.task.data.artifact_max_bytes, 6 * 1024 * 1024)

    async def test_setup_runtime_is_not_copied_or_serialized_and_finalize_releases_it(self):
        await self.task.setup(self.runtime, self.trace)
        self.assertEqual(set(self.runtime.files), {f"/workspace/{name}" for name in base.PUBLIC_FILES})
        cloned = copy.deepcopy(self.task)
        self.assertEqual(cloned.data, self.task.data)
        self.assertIs(adapter._SOLVERS[self.trace.id], self.runtime)
        self.assertNotIn("staged-fixture-runtime", self.trace.model_dump_json())
        await self.task.finalize(self.trace, self.runtime)
        self.assertNotIn(self.trace.id, adapter._SOLVERS)
        self.assertEqual(self.runtime.commands, [])

    async def test_abandoned_runtime_weak_reference_does_not_leak(self):
        runtime = RecordingRuntime()
        await self.task.setup(runtime, self.trace)
        del runtime
        gc.collect()
        self.assertNotIn(self.trace.id, adapter._SOLVERS)

    async def test_malformed_response_preserves_exact_arguments_and_scores_candidate_failure(self):
        session = self.session()
        malformed = ('{"command":"invalid\\ escape"}', '{"command":"ok"}}',
                     '{"command":"first","command":"second"}', '[]',
                     '{"command":"ok","x":NaN}', '{"command":"ok","x":1e999}')
        for arguments in malformed:
            with self.subTest(arguments=arguments):
                trace = self.new_trace(self.task)
                session = self.session(trace=trace)
                call = self.call("unused", arguments=arguments)
                await self.commit(session, [call])
                self.assertEqual(trace.nodes[-1].message.tool_calls[0].arguments, arguments)
                self.assertEqual(trace.stop_condition, "candidate_protocol_failure")
                self.assertEqual(await session.decide_tool(call.id, call.name),
                                 {"action": "stop", "reason": "candidate_protocol_failure"})
                await self.task.score(trace, self.runtime)
                self.assertEqual(trace.reward, 0)
                self.assertEqual(trace.metrics["candidate_failure"], 1)
                self.assertEqual(trace.errors, [])
                failure = trace.info[adapter.WORKFLOW_KEY]["protocol_failure"]
                self.assertEqual(failure["arguments_sha256"], hashlib.sha256(arguments.encode()).hexdigest())
                self.assertEqual(trace.info["financial_research_outcome"]["reason"], "malformed_tool_arguments")
        self.collect.assert_not_awaited()
        self.assertEqual(self.runtime.commands, [])

    async def test_mixed_malformed_batch_dispatches_no_call(self):
        session = self.session()
        calls = [self.call("printf harmless", "valid"), self.call("unused", "bad", arguments="{")]
        await self.commit(session, calls)
        self.assertEqual(session.tool_decisions, {})
        for call in calls:
            self.assertEqual((await session.decide_tool(call.id, call.name))["action"], "stop")
        self.collect.assert_not_awaited()

    async def test_ordinary_valid_command_is_allowed_without_checkpoint_or_execution(self):
        await self.task.setup(self.runtime, self.trace)
        session = self.session()
        call = self.call("printf ordinary")
        await self.commit(session, [call])
        self.assertEqual(await session.decide_tool(call.id, call.name, {"command": "printf ordinary"}),
                         {"action": "allow"})
        self.assertEqual(self.trace.info[adapter.WORKFLOW_KEY]["completed"], [])
        self.collect.assert_not_awaited()
        self.assertEqual(self.runtime.commands, [])

    async def test_checkpoint_denies_shell_execution_and_pins_accepted_result(self):
        await self.task.setup(self.runtime, self.trace)
        session, feedback = await self.checkpoint("ingestion")
        self.assertTrue(feedback["valid"])
        self.assertFalse(feedback["financial_correctness_checked"])
        self.assertEqual(feedback["next_stage"], "trials")
        self.assertEqual(self.trace.info[adapter.WORKFLOW_KEY]["completed"], ["ingestion"])
        original = self.trace.messages[-1]
        changed = original.model_copy(update={"content": "forged acceptance"})
        pinned = session.pin_tool_results(vf.Request(messages=[*self.trace.messages[:-1], changed]))
        self.assertEqual(pinned.messages[-1], original)
        self.assertEqual(self.collect.await_count, 2)
        for collected, path in zip(self.collect.await_args_list, adapter.PATHS[:2], strict=True):
            self.assertEqual([item.source for item in collected.args[1]], [path])
            self.assertEqual(collected.kwargs, {"max_bytes": adapter.LIMITS[path] + 65536, "sweep": False})
        self.assertEqual(self.runtime.commands, [])

    async def test_uncommitted_checkpoint_cannot_advance_workflow(self):
        await self.task.setup(self.runtime, self.trace)
        session = self.session()
        await self.commit(session, [self.call("printf normal")])
        decision = await session.decide_tool("uncommitted", "bash", {"command": "financial-checkpoint ingestion"})
        self.assertEqual(decision["action"], "deny")
        self.assertEqual(json.loads(decision["message"]["content"])["reason"], "checkpoint_not_committed")
        self.assertEqual(self.trace.info[adapter.WORKFLOW_KEY]["completed"], [])
        self.collect.assert_not_awaited()

    async def test_checkpoint_batch_and_out_of_order_are_denied(self):
        await self.task.setup(self.runtime, self.trace)
        session = self.session()
        await self.commit(session, [self.call("financial-checkpoint ingestion", "checkpoint"),
                                    self.call("printf normal", "normal")])
        decision = await session.decide_tool("checkpoint", "bash")
        self.assertEqual(json.loads(decision["message"]["content"])["reason"], "checkpoint_must_be_separate")
        trace = self.new_trace(self.task)
        await self.task.setup(self.runtime, trace)
        _, feedback = await self.checkpoint("trials", trace=trace)
        self.assertEqual(feedback["reason"], "checkpoint_out_of_order")
        self.collect.assert_not_awaited()

    async def test_missing_malformed_symlink_and_oversized_archives_cannot_advance(self):
        path = "/workspace/research.py"
        bad_archives = (None, b"not a tar", archive(path, kind=tarfile.SYMTYPE),
                        archive(path, b"x", name="tmp/foreign"),
                        archive(path, b"x" * (adapter.LIMITS[path] + 1)))
        for bad in bad_archives:
            with self.subTest(kind=type(bad).__name__, size=len(bad) if bad else 0):
                trace = self.new_trace(self.task)
                await self.task.setup(self.runtime, trace)
                artifacts = await self.collect_artifacts(self.runtime, self.task.data.artifacts)
                if bad is None:
                    artifacts[path] = None
                else:
                    artifacts[path] = bad
                with patch.object(adapter.vf, "collect", AsyncMock(return_value=artifacts)):
                    _, feedback = await self.checkpoint("ingestion", trace=trace)
                self.assertFalse(feedback["valid"])
                self.assertEqual(feedback["reason"], "missing_or_invalid_checkpoint_artifact")
                self.assertEqual(trace.info[adapter.WORKFLOW_KEY]["completed"], [])

    async def test_collection_failure_is_infrastructure_and_sanitized(self):
        await self.task.setup(self.runtime, self.trace)
        with patch.object(adapter.vf, "collect", AsyncMock(side_effect=RuntimeError("private-fixture-error"))):
            with self.assertRaisesRegex(Exception, "checkpoint_collection_failed") as error:
                await self.checkpoint("ingestion")
        self.assertNotIn("private-fixture-error", str(error.exception))
        self.assertEqual(self.trace.info["financial_research_outcome"]["status"], "infrastructure_error")
        self.assertEqual(self.trace.info[adapter.WORKFLOW_KEY]["completed"], [])

    async def test_checkpoint_transfer_limits_reject_oversized_ingestion_and_policy(self):
        for stage, path in (("ingestion", "/workspace/ingestion.json"),
                            ("submission", "/workspace/policy.json")):
            with self.subTest(stage=stage):
                trace = self.new_trace(self.task)
                await self.task.setup(self.runtime, trace)
                session = self.session(trace=trace)
                for preceding in adapter.STAGES[:adapter.STAGES.index(stage)]:
                    await self.checkpoint(preceding, trace=trace, session=session)
                completed = list(trace.info[adapter.WORKFLOW_KEY]["completed"])
                files_before = copy.deepcopy(self.runtime.files)
                payloads_before = copy.deepcopy(self.runtime.payloads)

                async def limited(runtime, artifacts, **kwargs):
                    if artifacts[0].source == path:
                        cap = adapter.LIMITS[path] + 65536
                        self.assertEqual(kwargs, {"max_bytes": cap, "sweep": False})
                        raise SandboxError(f"read '/tmp/vf-artifact-a012f9.tar': over the {cap} byte limit")
                    return await self.collect_artifacts(runtime, artifacts, **kwargs)

                with patch.object(adapter.vf, "collect", AsyncMock(side_effect=limited)):
                    _, feedback = await self.checkpoint(stage, trace=trace, session=session)
                self.assertFalse(feedback["valid"])
                self.assertEqual(feedback["reason"], "artifact_transfer_limit")
                self.assertEqual(trace.info[adapter.WORKFLOW_KEY]["completed"], completed)
                self.assertEqual(trace.errors, [])
                self.assertNotIn("financial_research_outcome", trace.info)
                self.assertEqual(self.runtime.files, files_before)
                self.assertEqual(self.runtime.payloads, payloads_before)
                self.assertEqual(self.runtime.commands, [])

    async def test_final_transfer_limit_is_accounted_as_candidate_zero(self):
        from financial_research.accounting import report_path

        records = []
        for path in ("/workspace/ingestion.json", "/workspace/policy.json"):
            with self.subTest(path=path):
                trace = self.new_trace(self.task)
                await self.task.setup(self.runtime, trace)
                session = self.session(trace=trace)
                for stage in adapter.STAGES:
                    await self.checkpoint(stage, trace=trace, session=session)
                files_before = copy.deepcopy(self.runtime.files)
                payloads_before = copy.deepcopy(self.runtime.payloads)
                cap = adapter.LIMITS[path] + 65536

                async def limited(runtime, artifacts, **kwargs):
                    if artifacts[0].source == path:
                        raise SandboxError(f"read '/tmp/vf-artifact-a012f9.tar': over the {cap} byte limit")
                    return await self.collect_artifacts(runtime, artifacts, **kwargs)

                with patch.object(adapter.vf, "collect", AsyncMock(side_effect=limited)) as collect:
                    await self.task.finalize(trace, self.runtime)
                self.assertEqual(collect.await_count, len(adapter.PATHS))
                self.assertEqual(set(trace.state.artifacts), set(adapter.PATHS))
                self.assertIsNone(trace.state.artifacts[path])
                self.assertTrue(trace.state.artifacts, "Native finalize must not repeat bulk collection")
                self.assertEqual(trace.info[adapter.WORKFLOW_KEY]["artifact_collection_violations"], [{
                    "path": path, "reason": "artifact_transfer_limit", "transfer_limit_bytes": cap,
                    "payload_saved": False}])
                self.assertNotIn(trace.id, adapter._SOLVERS)
                with patch.object(frozen.FinancialPortfolioTask, "_grade", AsyncMock()) as grade:
                    await self.task.score(trace, self.runtime)
                grade.assert_not_awaited()
                trace.stop("workflow_complete")
                trace.ok = True  # Native rollout seals this after successful finalization and scoring.
                self.assertEqual(trace.reward, 0)
                self.assertEqual(trace.metrics["candidate_failure"], 1)
                self.assertEqual(trace.errors, [])
                self.assertEqual(trace.info["financial_research_outcome"], {
                    "status": "candidate_failure", "reason": "artifact_transfer_limit",
                    "phase": "artifacts", "reproduction_verified": False})
                self.assertEqual(self.runtime.files, files_before)
                self.assertEqual(self.runtime.payloads, payloads_before)
                self.assertEqual(self.runtime.commands, [])
                records.append(trace.model_dump_json())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "traces.jsonl"
            output.write_text("\n".join(records) + "\n")
            report = report_path(output, expected=2)
        self.assertEqual(report["denominator"], 2)
        self.assertEqual(report["counts"]["candidate_failure"], 2, report)
        self.assertEqual(report["counts"]["infrastructure_error"], 0)
        self.assertEqual(report["counts"]["unknown"], 0)
        self.assertEqual(report["scored_attempts"], 2)
        self.assertEqual(report["mean_reward"], 0)
        self.assertEqual(report["issues"], [])
        self.assertTrue(report["performance_usable"])

    async def test_only_exact_native_transfer_limit_error_is_a_candidate_violation(self):
        cap = adapter.LIMITS["/workspace/research.py"] + 65536
        messages = ("transport unavailable private-fixture-error",
                    f"read '/tmp/other.tar': over the {cap} byte limit",
                    f"read '/tmp/vf-artifact-a012f9.tar': over the {cap + 1} byte limit")
        for phase in ("checkpoint", "finalize"):
            for message in messages:
                with self.subTest(phase=phase, message=message):
                    trace = self.new_trace(self.task)
                    await self.task.setup(self.runtime, trace)
                    with patch.object(adapter.vf, "collect", AsyncMock(side_effect=SandboxError(message))):
                        operation = (self.checkpoint("ingestion", trace=trace) if phase == "checkpoint"
                                     else self.task.finalize(trace, self.runtime))
                        reason = "checkpoint_collection_failed" if phase == "checkpoint" else "artifact_collection_failed"
                        with self.assertRaisesRegex(Exception, reason) as caught:
                            await operation
                    self.assertNotIn(message, str(caught.exception))
                    self.assertEqual(trace.info["financial_research_outcome"]["status"], "infrastructure_error")
                    self.assertEqual(trace.info["financial_research_outcome"]["reason"], reason)
                    self.assertFalse(trace.info[adapter.WORKFLOW_KEY].get("artifact_collection_violations"))
                    if phase == "finalize":
                        self.assertNotIn(trace.id, adapter._SOLVERS)

    async def test_malformed_surrogate_arguments_have_a_bounded_protocol_hash(self):
        arguments = '{"command":"' + chr(0xD800) + '"}}'
        response = self.response([self.call("unused", arguments=arguments)], self.trace)
        rewritten, records, stopped = await self.session().rewrite_response(response)
        self.assertEqual(rewritten.message.tool_calls[0].arguments, arguments)
        self.assertEqual(records, [])
        self.assertEqual(stopped, "candidate_protocol_failure")
        failure = self.trace.info[adapter.WORKFLOW_KEY]["protocol_failure"]
        self.assertEqual(failure["arguments_sha256"],
                         hashlib.sha256(arguments.encode("utf-8", "surrogatepass")).hexdigest())
        await self.task.score(self.trace, self.runtime)
        self.assertEqual(self.trace.reward, 0)
        self.assertEqual(self.trace.errors, [])

    async def test_concurrent_rollouts_collect_from_their_own_runtime(self):
        trace2 = self.new_trace(self.task)
        runtime2 = RecordingRuntime("second-runtime", {**self.payloads, "/workspace/research.py": b"not python !"})
        await asyncio.gather(self.task.setup(self.runtime, self.trace), self.task.setup(runtime2, trace2))
        first, second = await asyncio.gather(self.checkpoint("ingestion"), self.checkpoint("ingestion", trace=trace2))
        self.assertTrue(first[1]["valid"])
        self.assertFalse(second[1]["valid"])
        self.assertEqual(self.trace.info[adapter.WORKFLOW_KEY]["completed"], ["ingestion"])
        self.assertEqual(trace2.info[adapter.WORKFLOW_KEY]["completed"], [])
        self.assertEqual({id(call.args[0]) for call in self.collect.await_args_list}, {id(self.runtime), id(runtime2)})

    async def test_all_stages_gate_frozen_grader_and_final_mutation_fails(self):
        await self.task.setup(self.runtime, self.trace)
        session = self.session()
        trusted = {"reward": 0.25, "metrics": {"schema_valid": 1.0}}
        with patch.object(frozen.FinancialPortfolioTask, "_grade", AsyncMock(return_value=trusted)) as grade:
            for stage in adapter.STAGES:
                _, feedback = await self.checkpoint(stage, session=session)
                self.assertTrue(feedback["valid"])
                if stage != "submission":
                    self.assertEqual((await self.task._grade(self.trace, self.runtime))["reward"], 0)
                    grade.assert_not_awaited()
            self.assertTrue(await self.task.workflow_complete(self.trace))
            self.trace.state.artifacts = await self.collect_artifacts(self.runtime, self.task.data.artifacts)
            self.assertEqual(await self.task._grade(self.trace, self.runtime), trusted)
            grade.assert_awaited_once_with(self.trace, self.runtime)
            self.trace.state.artifacts["/workspace/research.py"] = archive("/workspace/research.py", b"x = 2\n")
            result = await self.task._grade(self.trace, self.runtime)
            self.assertEqual(result["reward"], 0)
            self.assertEqual(self.trace.info["financial_research_outcome"]["reason"], "checkpoint_artifact_changed")
            self.assertEqual(grade.await_count, 1)
        self.assertEqual(self.runtime.commands, [])

    async def test_native_isolated_grading_preserves_workflow_through_deepcopy(self):
        from verifiers.v1 import grading

        await self.task.setup(self.runtime, self.trace)
        session = self.session()
        for stage in adapter.STAGES:
            await self.checkpoint(stage, session=session)
        self.trace.state.artifacts = await self.collect_artifacts(self.runtime, self.task.data.artifacts)
        await self.task.finalize(self.trace, self.runtime)
        original_info = self.trace.info
        verifier = RecordingRuntime("isolated-verifier")
        verifier.env = {}
        verifier.with_env = lambda env: verifier
        verifier.backend = SimpleNamespace(prepare_setup=AsyncMock(), prepare_execution=AsyncMock())

        @asynccontextmanager
        async def provision(config, *, env):
            self.assertEqual(env, {})
            yield verifier

        deployment = SimpleNamespace(prepare_verifier=AsyncMock(), verifier=provision)
        seen = []

        async def frozen_grade(trace, runtime):
            self.assertIsNot(trace, self.trace)
            self.assertIsNot(trace.info, original_info)
            self.assertEqual(trace.info[adapter.WORKFLOW_KEY]["completed"], list(adapter.STAGES))
            self.assertIs(runtime, verifier)
            seen.append(trace)
            return {"reward": 0.25, "metrics": {"schema_valid": 1.0}}

        with patch.object(grading, "restore", AsyncMock()) as restore, patch.object(
                frozen.FinancialPortfolioTask, "_grade", AsyncMock(side_effect=frozen_grade)):
            await grading.grade_task(self.task, self.trace, vf.DockerConfig(), self.task.data.timeout,
                                     vf.ScoringConfig(mode="isolated", retries={"max_retries": 0}), deployment)
        self.assertTrue(seen)
        self.assertEqual(self.trace.reward, 0.25)
        self.assertEqual(self.trace.info[adapter.WORKFLOW_KEY]["completed"], list(adapter.STAGES))
        self.assertEqual(original_info[adapter.WORKFLOW_KEY]["completed"], list(adapter.STAGES))
        self.assertNotIn(self.trace.id, adapter._SOLVERS)
        restore.assert_awaited_once()
        self.assertEqual(verifier.commands, [])

    async def test_writable_state_does_not_complete_workflow(self):
        self.trace.state.grade = {"reward": 1, "metrics": {}}
        self.trace.state.__dict__[adapter.WORKFLOW_KEY] = {"completed": list(adapter.STAGES)}
        with patch.object(frozen.FinancialPortfolioTask, "_grade", AsyncMock()) as grade:
            result = await self.task._grade(self.trace, self.runtime)
        self.assertEqual(result["reward"], 0)
        self.assertEqual(self.trace.info["financial_research_outcome"]["reason"], "workflow_incomplete")
        grade.assert_not_awaited()

    async def test_plain_final_answer_stops_as_candidate_failure(self):
        session = self.session()
        await self.commit(session, [])
        self.assertEqual(self.trace.stop_condition, "candidate_workflow_failure")
        await self.task.score(self.trace, self.runtime)
        self.assertEqual(self.trace.reward, 0)
        self.assertEqual(self.trace.metrics["candidate_failure"], 1)
        self.assertEqual(self.trace.errors, [])

    async def test_modified_staged_instructions_rejected_before_private_staging(self):
        task = self.task.with_data(system_prompt="different workflow")
        with self.assertRaisesRegex(ValueError, "registered protocol"):
            await task.stage_verifier(self.trace, self.runtime)
        self.assertEqual(self.runtime.files, {})


if __name__ == "__main__":
    unittest.main()
