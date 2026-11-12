"""Native Verifiers tasks with trusted outputs and two isolated replays."""

from copy import deepcopy
import hashlib
import json

from pydantic import ConfigDict, Field
import verifiers.v1 as vf
from financial_research import prime_adapter as base
from financial_portfolio_staged.prime_adapter import _collect_bounded, _strict_object

from .cases import load_cases
from .reproduction import (
    MAX_POLICY_BYTES, MAX_SOURCE_BYTES, ReproductionInfrastructureError,
    outputs_match, parse_output, reproduce, valid_public_files,
)

PATHS = ("/workspace/research.py", "/workspace/policy.json")
PROTOCOL_ID = "financial-prerequisites-v1"
SYSTEM_PROMPT = (
    "Work in /workspace using the supplied public files and Python standard library. "
    "Save your complete working analysis as research.py, run python research.py, and leave "
    "the requested JSON object in policy.json. The same saved script must produce the correct "
    "output on alternate inputs with the same documented schema. Do not hardcode input values "
    "or outputs. Use the available bash tool to run commands and finish with a short final "
    "message after both artifacts exist. There are no checkpoint commands for these tasks."
)


class PrerequisiteData(vf.TaskData):
    model_config = ConfigDict(frozen=True, extra="forbid")
    public_files: dict[str, str]


class PrerequisiteState(vf.State):
    verifier_runtime: str | None = None
    grade: dict | None = None


class PrerequisiteTaskConfig(vf.TaskConfig):
    image: str = "python:3.12-slim"
    scoring_seconds: float = Field(default=180, gt=0, le=600)


class PrerequisiteTasksetConfig(vf.TasksetConfig):
    task: PrerequisiteTaskConfig = Field(default_factory=PrerequisiteTaskConfig)


def _record(trace):
    return trace.info.setdefault("prerequisite_check", {"protocol_id": PROTOCOL_ID})


class FinancialPrerequisiteTask(vf.Task[PrerequisiteData, PrerequisiteState, PrerequisiteTaskConfig]):
    NEEDS_CONTAINER = True

    def __init__(self, data, config=None):
        super().__init__(data, config)
        # Task attributes remain on the trusted worker; only TaskData is serialized
        # for the solver. No expected output or reference source enters that record.
        self._case = deepcopy(next(case for case in load_cases() if case["id"] == data.id))

    def grading_task(self, runtime, *, scoring):
        if scoring.mode != "isolated":
            raise ValueError("Prerequisites require isolated scoring")
        base._require_container(runtime)
        return self

    async def setup(self, runtime: vf.Runtime, trace: vf.Trace | None = None):
        base._require_container(runtime)
        if not valid_public_files(self.data.public_files):
            raise ValueError("Invalid public file set")
        try:
            for name, content in self.data.public_files.items():
                await runtime.write(f"/workspace/{name}", content.encode())
        except Exception:
            if trace is not None:
                base._outcome(trace, "infrastructure_error", "public_setup_failed", "setup", False)
            raise RuntimeError("public_setup_failed") from None

    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime):
        base._require_container(runtime)
        try:
            trace.state.artifacts, violations = await _collect_bounded(runtime, PATHS)
        except Exception:
            base._outcome(trace, "infrastructure_error", "artifact_collection_failed", "artifacts", False)
            raise RuntimeError("artifact_collection_failed") from None
        _record(trace)["artifact_collection_violations"] = violations

    async def stage_verifier(self, trace: vf.Trace, runtime: vf.Runtime):
        base._require_container(runtime)
        if (self.data.id != self._case["id"] or self.data.prompt != self._case["prompt"]
                or self.data.public_files != self._case["public_files"]
                or self.data.system_prompt != SYSTEM_PROMPT):
            raise RuntimeError("Task differs from its registered prerequisite")
        trace.state.verifier_runtime = runtime.name
        trace.state.grade = None
        _record(trace)["kind"] = self._case["kind"]

    @vf.stop(priority=100)
    async def candidate_protocol_failure(self, response: vf.Response, trace: vf.Trace) -> bool:
        for index, call in enumerate(response.message.tool_calls or []):
            try:
                _strict_object(call.arguments)
            except (ValueError, RecursionError):
                _record(trace)["protocol_failure"] = {
                    "reason": "malformed_tool_arguments", "tool_index": index,
                    "tool_name": call.name,
                    "arguments_sha256": hashlib.sha256(
                        call.arguments.encode("utf-8", "surrogatepass")).hexdigest(),
                }
                return True
        return False

    def _result(self, trace, status, reason, phase, *, analysis=False, submitted=False,
                original=False, alternate=False):
        verified = original and alternate
        base._outcome(trace, status, reason, phase, verified)
        trace.state.grade = {"reward": float(status == "passed"), "metrics": {
            "analysis_artifact_valid": float(analysis), "submitted_output_correct": float(submitted),
            "original_reproduction_passed": float(original),
            "alternate_reproduction_passed": float(alternate),
            "reproduction_verified": float(verified), "candidate_failure": float(status == "candidate_failure"),
        }}
        return trace.state.grade

    async def _grade(self, trace: vf.Trace, runtime: vf.Runtime):
        try:
            return await self._grade_checked(trace, runtime)
        except Exception:
            if trace.info.get("financial_research_outcome", {}).get("status") != "infrastructure_error":
                base._outcome(trace, "infrastructure_error", "scoring_error", "scoring", False)
            raise RuntimeError(trace.info["financial_research_outcome"]["reason"]) from None

    async def _grade_checked(self, trace, runtime):
        base._require_container(runtime)
        if trace.state.verifier_runtime != runtime.name:
            raise RuntimeError("Scoring requires the prepared isolated verifier")
        if trace.state.grade is not None:
            return trace.state.grade
        record = _record(trace)
        if record.get("protocol_failure"):
            return self._result(trace, "candidate_failure", "malformed_tool_arguments", "protocol")
        if record.get("artifact_collection_violations"):
            return self._result(trace, "candidate_failure", "artifact_transfer_limit", "artifacts")
        analysis_valid = False
        try:
            source = base._artifact_bytes(trace.state.artifacts, PATHS[0], MAX_SOURCE_BYTES)
            try:
                if not source.decode("utf-8").strip():
                    raise base.CandidateArtifactError("invalid_analysis_artifact")
            except UnicodeError:
                raise base.CandidateArtifactError("invalid_analysis_artifact") from None
            analysis_valid = True
            payload = base._artifact_bytes(trace.state.artifacts, PATHS[1], MAX_POLICY_BYTES)
            try:
                submitted = parse_output(payload)
            except (ValueError, UnicodeError, RecursionError):
                raise base.CandidateArtifactError("invalid_policy_json") from None
        except base.CandidateArtifactError as error:
            return self._result(trace, "candidate_failure", str(error), "artifacts", analysis=analysis_valid)
        if not outputs_match(submitted, self._case["expected_json"]):
            return self._result(trace, "candidate_failure", "submitted_output_mismatch", "scoring", analysis=True)
        passed = {}
        for label, files_key, expected_key in (
            ("original", "public_files", "expected_json"),
            ("alternate", "alternate_public_files", "alternate_expected_json"),
        ):
            try:
                reason = await reproduce(self._case[files_key], source, self._case[expected_key],
                                         runtime_config=runtime.config)
            except ReproductionInfrastructureError as error:
                base._outcome(trace, "infrastructure_error", str(error), f"{label}_reproduction", False)
                raise
            record[f"{label}_reproduction"] = reason
            if reason != "passed":
                return self._result(trace, "candidate_failure", reason, f"{label}_reproduction",
                                    analysis=True, submitted=True, **passed)
            passed[label] = True
        return self._result(trace, "passed", "prerequisite_completed", "scoring", analysis=True,
                            submitted=True, **passed)

    @vf.metric
    async def prerequisite_metrics(self, trace: vf.Trace, runtime: vf.Runtime) -> dict[str, float]:
        return (await self._grade(trace, runtime))["metrics"]

    @vf.reward
    async def prerequisite_reward(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        return (await self._grade(trace, runtime))["reward"]

    async def validate(self, trace: vf.Trace, runtime: vf.Runtime) -> bool:
        _record(trace)["model_free_reference"] = True
        await runtime.write(PATHS[0], self._case["reference_source"].encode())
        await runtime.write(PATHS[1], json.dumps(self._case["expected_json"], allow_nan=False).encode())
        await self.finalize(trace, runtime)
        await vf.grade_submission(self, trace, runtime)
        return trace.reward == 1.0


class FinancialPrerequisitesTaskset(vf.Taskset[FinancialPrerequisiteTask, PrerequisiteTasksetConfig]):
    def load(self):
        for case in load_cases():
            data = PrerequisiteData(
                id=case["id"], prompt=case["prompt"], system_prompt=SYSTEM_PROMPT,
                public_files=case["public_files"], image=self.config.task.image, workdir="/workspace",
                network=vf.NetworkPolicyConfig(allow=[]),
                scoring=vf.ScoringConfig(mode="isolated", network=vf.NetworkPolicyConfig(allow=[])),
                artifacts=[vf.Artifact(source=path, required=False) for path in PATHS],
                artifact_max_bytes=4 * 1024 * 1024,
                timeout=vf.TaskTimeout(setup=120, agent=600, finalize=30,
                                       scoring=self.config.task.scoring_seconds),
                resources=vf.TaskResources(cpu=1, memory=1),
            )
            yield FinancialPrerequisiteTask(data, self.config.task)
