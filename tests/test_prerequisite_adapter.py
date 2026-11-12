"""Boundary and accounting regressions for the prerequisite diagnostic."""

import base64
from contextlib import asynccontextmanager
import io
import json
import os
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
for package in ("financial_research", "financial_portfolio", "financial_portfolio_staged",
                "financial_prerequisites"):
    sys.path.insert(0, str(ROOT / "environments" / package / "src"))
from financial_prerequisites.cases import load_cases
from financial_prerequisites import reproduction
try:
    import verifiers.v1 as vf
except ModuleNotFoundError as error:
    if error.name != "verifiers":
        raise
    vf = None
if vf is not None:
    from financial_prerequisites import prime_adapter as adapter


def archive(path, payload, kind=tarfile.REGTYPE):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as bundle:
        entry = tarfile.TarInfo(path.lstrip("/"))
        entry.type = kind
        entry.size = len(payload) if kind == tarfile.REGTYPE else 0
        bundle.addfile(entry, io.BytesIO(payload) if kind == tarfile.REGTYPE else None)
    return buffer.getvalue()


class OutputContract(unittest.TestCase):
    def test_strict_integer_null_bool_shape_and_order(self):
        expected = {"n": 1, "value": 1.5, "missing": None, "flag": True, "ids": ["a", "b"]}
        self.assertTrue(reproduction.outputs_match(expected, expected))
        for update in ({"n": True}, {"n": 1.0}, {"missing": 0}, {"flag": 1},
                       {"ids": ["b", "a"]}, {"extra": 1}):
            with self.subTest(update=update):
                self.assertFalse(reproduction.outputs_match({**expected, **update}, expected))
        self.assertFalse(reproduction.outputs_match({"n": 1}, expected))
        self.assertTrue(reproduction.outputs_match(1, 1.0))
        self.assertFalse(reproduction.outputs_match(True, 1.0))

    def test_numeric_tolerance_and_extremes(self):
        self.assertTrue(reproduction.outputs_match(0.5 + 5e-10, 0.5))
        self.assertFalse(reproduction.outputs_match(0.5 + 2e-9, 0.5))
        for value in (float("nan"), float("inf"), 10 ** 1000):
            self.assertFalse(reproduction.outputs_match(value, 1.0))

    def test_strict_json_parser(self):
        for payload in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}',
                        b'{"x":[1e999]}', b'[]', b'null', b'{"x":1,}'):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                reproduction.parse_output(payload)
        self.assertEqual(reproduction.parse_output(b'{"n":1,"x":null}'), {"n": 1, "x": None})

    def test_public_file_paths_cannot_overwrite_artifacts_or_controller(self):
        for name in ("../outside", "/tmp/outside", "research.py", "policy.json", "..", "a/b"):
            self.assertFalse(reproduction.valid_public_files({name: "x"}))
        self.assertTrue(reproduction.valid_public_files({"starter.py": "x", "contract.json": "{}"}))


@unittest.skipUnless(vf is not None, "Install pinned optional Verifiers dependencies")
class PrerequisiteAdapter(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.task = next(iter(adapter.FinancialPrerequisitesTaskset(adapter.PrerequisiteTasksetConfig())))
        self.case = load_cases()[0]
        self.runtime = SimpleNamespace(type="docker", config=vf.DockerConfig(), env={},
                                       name="isolated-verifier", write=AsyncMock())
        self.trace = vf.Trace(
            task=vf.TraceTask(type=type(self.task).__name__, data=self.task.data),
            agent=vf.AgentInfo(config=vf.AgentConfig(runtime=vf.DockerConfig())),
            state=adapter.PrerequisiteState())

    async def stage(self, source=None, output=None):
        source = (source if source is not None else self.case["reference_source"]).encode()
        output = self.case["expected_json"] if output is None else output
        self.trace.state.artifacts = {
            adapter.PATHS[0]: archive(adapter.PATHS[0], source),
            adapter.PATHS[1]: archive(adapter.PATHS[1], json.dumps(output).encode()),
        }
        await self.task.stage_verifier(self.trace, self.runtime)

    def test_task_wire_has_no_trusted_case_fields(self):
        wire = self.task.data.model_dump_json()
        for field in ("expected_json", "alternate_public_files", "alternate_expected_json", "reference_source"):
            self.assertNotIn(field, self.task.data.model_dump())
        self.assertNotIn(self.case["reference_source"], wire)
        self.assertEqual(len(list(adapter.FinancialPrerequisitesTaskset(adapter.PrerequisiteTasksetConfig()))), 4)

    async def test_setup_stages_only_original_public_files(self):
        await self.task.setup(self.runtime, self.trace)
        self.assertEqual({call.args[0]: call.args[1] for call in self.runtime.write.call_args_list},
                         {f"/workspace/{name}": text.encode() for name, text in self.case["public_files"].items()})

    async def test_modified_public_record_is_rejected_before_scoring(self):
        self.task.data = self.task.data.model_copy(update={"public_files": {"contract.json": "{}"}})
        with self.assertRaisesRegex(RuntimeError, "registered prerequisite"):
            await self.task.stage_verifier(self.trace, self.runtime)

    async def test_success_requires_original_and_alternate_replay(self):
        await self.stage()
        with patch.object(adapter, "reproduce", AsyncMock(return_value="passed")) as replay:
            grade = await self.task._grade(self.trace, self.runtime)
            self.assertEqual(grade["reward"], 1)
            self.assertEqual(replay.await_count, 2)
            self.assertEqual(replay.await_args_list[0].args[0], self.case["public_files"])
            self.assertEqual(replay.await_args_list[1].args[0], self.case["alternate_public_files"])
            self.assertEqual(grade["metrics"]["alternate_reproduction_passed"], 1)
            self.assertTrue(self.trace.info["financial_research_outcome"]["reproduction_verified"])
            self.assertEqual(self.trace.info["prerequisite_check"]["alternate_reproduction"], "passed")
            await self.task._grade(self.trace, self.runtime)
            self.assertEqual(replay.await_count, 2, "Metric and reward reuse one grade")

    async def test_alternate_mismatch_is_zero_candidate_failure(self):
        await self.stage()
        with patch.object(adapter, "reproduce", AsyncMock(side_effect=["passed", "output_mismatch"])):
            grade = await self.task._grade(self.trace, self.runtime)
        self.assertEqual(grade["reward"], 0)
        self.assertEqual(grade["metrics"]["original_reproduction_passed"], 1)
        self.assertEqual(grade["metrics"]["alternate_reproduction_passed"], 0)
        self.assertEqual(self.trace.info["financial_research_outcome"], {
            "status": "candidate_failure", "reason": "output_mismatch",
            "phase": "alternate_reproduction", "reproduction_verified": False})

    async def test_wrong_submission_does_not_launch_replay(self):
        await self.stage(output={})
        with patch.object(adapter, "reproduce", AsyncMock()) as replay:
            grade = await self.task._grade(self.trace, self.runtime)
        replay.assert_not_awaited()
        self.assertEqual(grade["reward"], 0)
        self.assertEqual(self.trace.info["financial_research_outcome"]["reason"], "submitted_output_mismatch")

    async def test_missing_and_unsafe_artifact_are_candidate_failures(self):
        for payload in (None, archive(adapter.PATHS[0], b"", tarfile.SYMTYPE), b"not a tar"):
            await self.stage()
            self.trace.state.artifacts[adapter.PATHS[0]] = payload
            with patch.object(adapter, "reproduce", AsyncMock()) as replay:
                grade = await self.task._grade(self.trace, self.runtime)
            replay.assert_not_awaited()
            self.assertEqual(grade["reward"], 0)
            self.assertEqual(self.trace.info["financial_research_outcome"]["status"], "candidate_failure")

    async def test_protocol_failure_preserves_zero_grade(self):
        await self.stage()
        response = vf.Response(id="fixture-response", created=0, model="fixture", finish_reason="tool_calls",
            message=vf.AssistantMessage(tool_calls=[
            vf.ToolCall(id="bad", name="bash", arguments='{"command":"invalid\\ escape"}')]))
        self.assertTrue(await self.task.candidate_protocol_failure(response, self.trace))
        with patch.object(adapter, "reproduce", AsyncMock()) as replay:
            grade = await self.task._grade(self.trace, self.runtime)
        replay.assert_not_awaited()
        self.assertEqual(grade["reward"], 0)
        self.assertEqual(self.trace.info["financial_research_outcome"]["reason"], "malformed_tool_arguments")

    async def test_infrastructure_failure_is_not_candidate_zero_and_message_is_sanitized(self):
        await self.stage()
        with patch.object(adapter, "reproduce", AsyncMock(side_effect=
                          reproduction.ReproductionInfrastructureError("reproduction_service_failed"))):
            with self.assertRaisesRegex(RuntimeError, "^reproduction_service_failed$"):
                await self.task._grade(self.trace, self.runtime)
        self.assertIsNone(self.trace.state.grade)
        self.assertEqual(self.trace.info["financial_research_outcome"]["status"], "infrastructure_error")
        self.assertEqual(self.trace.info["scoring_diagnostics"]["financial_research_outcome"],
                         self.trace.info["financial_research_outcome"])

    async def test_stage_verifier_resets_untrusted_grade(self):
        self.trace.state.grade = {"reward": 1.0}
        await self.task.stage_verifier(self.trace, self.runtime)
        self.assertIsNone(self.trace.state.grade)


@unittest.skipUnless(vf is not None, "Install pinned optional Verifiers dependencies")
class ReplayBoundary(unittest.IsolatedAsyncioTestCase):
    async def test_only_source_and_public_inputs_enter_fresh_runtime(self):
        case = load_cases()[0]
        writes = {}
        runtime = SimpleNamespace(
            backend=SimpleNamespace(prepare_setup=AsyncMock(), prepare_execution=AsyncMock()),
            write=AsyncMock(side_effect=lambda path, content: writes.update({path: content})),
            run=AsyncMock(return_value=SimpleNamespace(exit_code=0)),
            read=AsyncMock(return_value=json.dumps({"reason": "output_created", "policy":
                base64.b64encode(json.dumps(case["expected_json"]).encode()).decode()}).encode()))
        configurations = []

        @asynccontextmanager
        async def provision(config, env):
            configurations.append((config, env))
            yield runtime

        with patch("verifiers.v1.runtimes.provision_runtime", provision):
            reason = await reproduction.reproduce(case["public_files"], case["reference_source"].encode(),
                case["expected_json"], runtime_config=vf.DockerConfig(allow=["example.com"]))
        self.assertEqual(reason, "passed")
        self.assertEqual(configurations[0][1], {})
        self.assertEqual(configurations[0][0].allow, [])
        self.assertEqual(configurations[0][0].mounts, {})
        self.assertEqual(set(writes), {"/opt/financial-reproduction/controller.py", "/workspace/research.py",
                                      *(f"/workspace/{name}" for name in case["public_files"])})
        self.assertNotIn("/workspace/policy.json", writes)
        self.assertEqual(runtime.run.await_args_list[-1].args[0][2],
                         "/opt/financial-reproduction/controller.py")

    async def test_provider_exception_details_do_not_escape(self):
        case = load_cases()[0]

        @asynccontextmanager
        async def provision(config, env):
            raise RuntimeError("fixture-sensitive-provider-detail")
            yield

        with patch("verifiers.v1.runtimes.provision_runtime", provision):
            with self.assertRaisesRegex(reproduction.ReproductionInfrastructureError,
                                        "^reproduction_service_failed$") as caught:
                await reproduction.reproduce(case["public_files"], case["reference_source"].encode(),
                                             case["expected_json"], runtime_config=vf.DockerConfig())
        self.assertTrue(caught.exception.__suppress_context__)


@unittest.skipUnless(vf is not None and os.environ.get("RUN_PREREQUISITE_DOCKER_TESTS") == "1",
                     "Enable RUN_PREREQUISITE_DOCKER_TESTS=1 with Docker installed")
class DockerReplay(unittest.IsolatedAsyncioTestCase):
    async def test_hardcoded_original_answer_fails_alternate_in_fresh_container(self):
        case = load_cases()[0]
        source = ("import json\nwith open('policy.json','w') as f:\n"
                  "    json.dump(" + repr(case["expected_json"]) + ",f)\n").encode()
        for label, files_key, expected_key in (("original", "public_files", "expected_json"),
                ("alternate", "alternate_public_files", "alternate_expected_json")):
            result = await reproduction.reproduce(case[files_key], source, case[expected_key],
                                                  runtime_config=vf.DockerConfig(image="python:3.12-slim"))
            self.assertEqual(result, "passed" if label == "original" else "output_mismatch")

    async def test_script_exit_is_candidate_failure_in_fresh_container(self):
        case = load_cases()[0]
        result = await reproduction.reproduce(case["public_files"], b"raise SystemExit(3)\n",
            case["expected_json"], runtime_config=vf.DockerConfig(image="python:3.12-slim"))
        self.assertEqual(result, "analysis_exit")


if __name__ == "__main__":
    unittest.main()
