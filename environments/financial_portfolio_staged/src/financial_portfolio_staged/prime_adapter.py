"""Trusted staged checkpoints around the unchanged portfolio financial grader."""

from __future__ import annotations

import hashlib
from importlib.resources import files
import json
import math
import re
from weakref import WeakValueDictionary

import verifiers.v1 as vf
from verifiers.v1.errors import SandboxError
from financial_portfolio.prime_adapter import FinancialPortfolioTask, PortfolioTasksetConfig
from financial_research import prime_adapter as base

from .validation import ARTIFACT_LIMITS, inventory_for_public_files, stage_contract, validate_stage


PROTOCOL_ID = "portfolio-staged-v1"
STAGES = ("ingestion", "trials", "submission")
WORKFLOW_KEY = "portfolio_workflow"
PATHS = ("/workspace/research.py", "/workspace/ingestion.json",
         "/workspace/trials.json", "/workspace/policy.json")
LIMITS = ARTIFACT_LIMITS
# Runtime objects must not enter Task deepcopy or the serialized trace. Weak values
# also release abandoned rollouts whose finalization hook could not run.
_SOLVERS: WeakValueDictionary[str, vf.Runtime] = WeakValueDictionary()


def workflow_prompt(public_files: dict[str, str]) -> str:
    instructions = files("financial_portfolio_staged").joinpath("instructions.txt").read_text()
    return instructions + json.dumps(stage_contract(public_files), sort_keys=True, indent=2)


def _strict_object(arguments: str) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate_key")
            result[key] = value
        return result

    def reject(value):
        raise ValueError("nonfinite_number")

    value = json.loads(arguments, object_pairs_hook=unique, parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError("arguments_not_object")
    # parse_constant does not reject overflowed finite JSON literals such as 1e999.
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
        elif isinstance(item, float) and not math.isfinite(item):
            raise ValueError("nonfinite_number")
    return value


def _payloads(archives: dict, paths=PATHS) -> dict[str, bytes]:
    return {path: base._artifact_bytes(archives, path, LIMITS[path]) for path in paths}


def _hashes(payloads: dict[str, bytes]) -> dict[str, str]:
    return {path: hashlib.sha256(value).hexdigest() for path, value in payloads.items()}


async def _collect_bounded(runtime: vf.Runtime, paths=PATHS) -> tuple[dict, list[dict]]:
    """Collect declared files individually; retain size violations as candidate data.

    Native finalization uses precollected artifacts when supplied by the task.
    Oversized payloads stay in the solver, with an explicit omission record, rather
    than turning a candidate limit violation into a native infrastructure error.
    """
    archives, violations = {}, []
    for path in paths:
        cap = LIMITS[path] + 65536  # Space for the bounded regular-file tar envelope.
        try:
            captured = await vf.collect(runtime, [vf.Artifact(source=path, required=False)],
                                        max_bytes=cap, sweep=False)
            archives[path] = captured[path]
        except SandboxError as error:
            if not re.fullmatch(r"read '/tmp/vf-artifact-[0-9a-f]+\.tar': over the "
                                + str(cap) + r" byte limit", str(error)):
                raise
            archives[path] = None
            violations.append({"path": path, "reason": "artifact_transfer_limit",
                               "transfer_limit_bytes": cap, "payload_saved": False})
    return archives, violations


def _workflow(trace: vf.Trace) -> dict:
    # info is owned by the coordinator; it is not the tool servers' writable State.
    return trace.info.setdefault(WORKFLOW_KEY, {
        "protocol_id": PROTOCOL_ID, "completed": [], "checkpoints": [],
    })


class FinancialPortfolioStagedTask(FinancialPortfolioTask):
    async def setup(self, runtime: vf.Runtime, trace: vf.Trace | None = None) -> None:
        await super().setup(runtime, trace)
        if trace is not None:
            _SOLVERS[trace.id] = runtime
            _workflow(trace)

    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        try:
            await super().finalize(trace, runtime)
            try:
                trace.state.artifacts, violations = await _collect_bounded(runtime)
            except Exception:
                base._outcome(trace, "infrastructure_error", "artifact_collection_failed", "artifacts", False)
                raise RuntimeError("artifact_collection_failed") from None
            _workflow(trace)["artifact_collection_violations"] = violations
        finally:
            _SOLVERS.pop(trace.id, None)

    async def stage_verifier(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        _SOLVERS.pop(trace.id, None)
        if self.data.system_prompt != workflow_prompt(self.data.public_files):
            raise ValueError("Staged system instructions differ from the registered protocol")
        await super().stage_verifier(trace, runtime)

    @vf.stop(priority=100)
    async def candidate_protocol_failure(self, response: vf.Response, trace: vf.Trace) -> bool:
        for index, call in enumerate(response.message.tool_calls or []):
            try:
                _strict_object(call.arguments)
            except (ValueError, RecursionError):
                _workflow(trace)["protocol_failure"] = {
                    "reason": "malformed_tool_arguments", "tool_index": index,
                    "tool_name": call.name,
                    "arguments_sha256": hashlib.sha256(call.arguments.encode("utf-8", "surrogatepass")).hexdigest(),
                }
                return True
        return False

    @vf.stop
    async def candidate_workflow_failure(self, response: vf.Response, trace: vf.Trace) -> bool:
        return (not response.message.tool_calls
                and _workflow(trace)["completed"] != list(STAGES))

    @vf.stop
    async def workflow_complete(self, trace: vf.Trace) -> bool:
        return _workflow(trace)["completed"] == list(STAGES)

    @vf.intercept
    async def checkpoint(self, request: vf.Request, trace: vf.Trace) -> vf.Request | None:
        if not request.messages or not isinstance(request.messages[-1], vf.ToolMessage):
            return None
        result = request.messages[-1]
        if result.content != "":
            return None
        assistant = next((m for m in reversed(request.messages[:-1])
                          if isinstance(m, vf.AssistantMessage)), None)
        if assistant is None:
            return None
        call = next((c for c in assistant.tool_calls or [] if c.id == result.tool_call_id), None)
        if call is None or call.name != "bash":
            return None
        try:
            arguments = _strict_object(call.arguments)
        except (ValueError, RecursionError):
            return None  # The response stop rejects this before tool dispatch.
        command = arguments.get("command")
        if not isinstance(command, str) or command.strip() not in {
                f"financial-checkpoint {stage}" for stage in STAGES}:
            return None
        stage = command.strip().split()[1]
        workflow = _workflow(trace)
        committed = next((node.message for node in reversed(trace.nodes)
                          if isinstance(node.message, vf.AssistantMessage)), None)
        payloads = None
        if assistant != committed:
            verdict = {"valid": False, "reason": "checkpoint_not_committed"}
        elif len(assistant.tool_calls) != 1:
            verdict = {"valid": False, "reason": "checkpoint_must_be_separate"}
        elif len(workflow["completed"]) >= len(STAGES) or stage != STAGES[len(workflow["completed"])]:
            verdict = {"valid": False, "reason": "checkpoint_out_of_order"}
        else:
            runtime = _SOLVERS.get(trace.id)
            if runtime is None:
                base._outcome(trace, "infrastructure_error", "checkpoint_runtime_missing", "checkpoint", False)
                raise RuntimeError("checkpoint_runtime_missing")
            paths = PATHS[:STAGES.index(stage) + 2]
            try:
                archives, violations = await _collect_bounded(runtime, paths)
            except Exception:
                base._outcome(trace, "infrastructure_error", "checkpoint_collection_failed", "checkpoint", False)
                raise RuntimeError("checkpoint_collection_failed") from None
            try:
                payloads = _payloads(archives, paths)
                verdict = validate_stage(stage, self.data.public_files, payloads)
            except base.CandidateArtifactError:
                verdict = {"valid": False, "reason": "artifact_transfer_limit" if violations
                           else "missing_or_invalid_checkpoint_artifact"}
        record = {"stage": stage, **verdict, "assistant_node": len(trace.nodes) - 1}
        if verdict["valid"]:
            workflow["completed"].append(stage)
            record["artifact_sha256"] = _hashes(payloads)
            if stage == "submission":
                workflow["submission_sha256"] = record["artifact_sha256"]
        workflow["checkpoints"].append(record)
        next_stage = STAGES[len(workflow["completed"])] if len(workflow["completed"]) < 3 else None
        feedback = {"checkpoint": stage, **verdict, "next_stage": next_stage,
                    "financial_correctness_checked": False}
        replacement = result.model_copy(update={"content": json.dumps(feedback, sort_keys=True)})
        return request.model_copy(update={"messages": [*request.messages[:-1], replacement]})

    def _workflow_failure_grade(self, trace: vf.Trace, reason: str, phase="workflow") -> dict:
        base._outcome(trace, "candidate_failure", reason, phase, False)
        trace.state.grade = base._failure_grade()
        return trace.state.grade

    async def _grade(self, trace: vf.Trace, runtime: vf.Runtime) -> dict:
        workflow = _workflow(trace)
        if workflow.get("protocol_failure"):
            return self._workflow_failure_grade(trace, "malformed_tool_arguments", "protocol")
        if workflow.get("artifact_collection_violations"):
            return self._workflow_failure_grade(trace, "artifact_transfer_limit", "artifacts")
        if workflow["completed"] != list(STAGES):
            return self._workflow_failure_grade(trace, "workflow_incomplete")
        try:
            payloads = _payloads(trace.state.artifacts)
        except base.CandidateArtifactError:
            return self._workflow_failure_grade(trace, "invalid_final_checkpoint_artifacts")
        if _hashes(payloads) != workflow.get("submission_sha256"):
            return self._workflow_failure_grade(trace, "checkpoint_artifact_changed")
        verdict = validate_stage("submission", self.data.public_files, payloads)
        if not verdict["valid"]:
            return self._workflow_failure_grade(trace, "invalid_final_checkpoint_artifacts")
        return await super()._grade(trace, runtime)

    async def validate(self, trace: vf.Trace, runtime: vf.Runtime) -> bool:
        """Model-free positive control, independently labelled by vf-validate."""
        from financial_portfolio.generator import resolve_episode
        from financial_research.verifier import reference_submission, score_submission

        episode = resolve_episode(self.data.id)
        policy = reference_submission(episode)
        expected = score_submission(episode, policy)["reward"]
        payloads = {
            PATHS[0]: files("financial_portfolio").joinpath("reference.py").read_bytes(),
            PATHS[1]: json.dumps(inventory_for_public_files(self.data.public_files)).encode(),
            PATHS[2]: json.dumps(policy["trials"], allow_nan=False).encode(),
            PATHS[3]: json.dumps(policy, allow_nan=False).encode(),
        }
        workflow = _workflow(trace)
        for stage in STAGES:
            if not validate_stage(stage, self.data.public_files, payloads)["valid"]:
                return False
            workflow["completed"].append(stage)
        workflow["submission_sha256"] = _hashes(payloads)
        workflow["model_free_reference"] = True
        for path, content in payloads.items():
            await runtime.write(path, content)
        await self.finalize(trace, runtime)
        await vf.grade_submission(self, trace, runtime)
        return expected > 0 and math.isclose(trace.reward, expected, abs_tol=1e-12)


class FinancialPortfolioStagedTaskset(vf.Taskset[FinancialPortfolioStagedTask, PortfolioTasksetConfig]):
    def load(self):
        from financial_portfolio.prime_adapter import FinancialPortfolioTaskset

        for task in FinancialPortfolioTaskset(self.config).load():
            data = task.data.model_copy(update={
                "system_prompt": workflow_prompt(task.data.public_files),
                "artifacts": [vf.Artifact(source=path, required=False) for path in PATHS],
                "artifact_max_bytes": 6 * 1024 * 1024,
            })
            yield FinancialPortfolioStagedTask(data, task.config)
