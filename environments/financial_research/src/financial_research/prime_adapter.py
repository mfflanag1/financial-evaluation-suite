"""Native Verifiers v1 taskset with public solver data and isolated grading.

The supported API is ``verifiers==0.4.1``. TaskData is trace-visible: it must
never carry generator seeds, family labels, answers, or hidden continuations.
Only the trusted worker resolves an opaque episode identifier for grading.
"""

from __future__ import annotations

import asyncio
import io
import json
import math
import tarfile
from importlib.resources import files
from typing import Any, Literal
from uuid import uuid4

from pydantic import ConfigDict, Field, field_validator, model_validator
import verifiers.v1 as vf


PUBLIC_FILES = frozenset({
    "prices.csv", "features.csv", "securities.csv", "contract.json",
})
MAX_POLICY_BYTES = 2 * 1024 * 1024
MAX_ANALYSIS_BYTES = 1024 * 1024

# This program is trusted package code. It parses a declarative policy as data;
# research.py is never imported or executed in the verifier runtime.
_SCORE_PROGRAM = r'''
import json
import os
from pathlib import Path
import stat
import sys

root = Path(__file__).parent
sys.path.insert(0, str(root))
from financial_research.verifier import score_submission

def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate object key")
        result[key] = value
    return result

def reject_constant(value):
    raise ValueError("non-finite JSON number")

def valid_analysis_artifact():
    try:
        fd = os.open("/workspace/research.py", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as error:
        import errno
        if error.errno in (errno.ENOENT, errno.ELOOP, errno.EACCES, errno.ENOTDIR):
            return False
        raise
    with os.fdopen(fd, "rb") as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode) or not 0 < metadata.st_size <= 1048576:
            return False
        payload = stream.read(1048577)
    if len(payload) > 1048576:
        return False
    try:
        return bool(payload.decode("utf-8").strip())
    except UnicodeError:
        return False

def read_submission():
    path = "/workspace/policy.json"
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except (FileNotFoundError, IsADirectoryError):
        return None
    except OSError as error:
        import errno
        if error.errno in (errno.ELOOP, errno.EACCES, errno.ENOTDIR):
            return None
        raise
    with os.fdopen(fd, "rb") as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 2097152:
            return None
        payload = stream.read(2097153)
    if len(payload) > 2097152:
        return None
    try:
        return json.loads(payload, object_pairs_hook=unique_object,
                          parse_constant=reject_constant)
    except (ValueError, UnicodeError, RecursionError):
        return None

episode = json.loads((root / "episode.json").read_text())
result = score_submission(episode, read_submission())
present = valid_analysis_artifact()
result["metrics"]["analysis_artifact_valid"] = float(present)
if not present:
    result["reward"] = 0.0
# Do not return grader diagnostics or hidden values in a rollout trace.
print(json.dumps({"reward": result["reward"], "metrics": result["metrics"]},
                 allow_nan=False, separators=(",", ":")))
'''


class ResearchData(vf.TaskData):
    """The complete wire record, containing only solver-visible information."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    public_files: dict[str, str]


class ResearchState(vf.State):
    verifier_directory: str | None = None
    verifier_runtime: str | None = None
    grade: dict[str, Any] | None = None
    reproduction: dict[str, Any] | None = None
    analysis_artifact_valid: bool = False


class ResearchTaskConfig(vf.TaskConfig):
    image: str = "python:3.12-slim"
    scoring_seconds: float = Field(default=180, gt=0, le=600)


class FinancialResearchTask(vf.Task[ResearchData, ResearchState, ResearchTaskConfig]):
    NEEDS_CONTAINER = True

    def grading_task(self, runtime: vf.Runtime, *, scoring: vf.ScoringConfig):
        if scoring.mode != "isolated":
            raise ValueError("Financial research tasks require isolated scoring")
        _require_container(runtime)
        return self

    async def setup(self, runtime: vf.Runtime, trace: vf.Trace | None = None) -> None:
        _require_container(runtime)
        if set(self.data.public_files) != PUBLIC_FILES:
            raise ValueError("Unexpected public file set")
        try:
            for name, content in self.data.public_files.items():
                await runtime.write(f"/workspace/{name}", content.encode("utf-8"))
        except Exception:
            if trace is not None:
                _outcome(trace, "infrastructure_error", "public_setup_failed", "setup", False)
            raise RuntimeError("public_setup_failed") from None

    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        _require_container(runtime)
        trace.info["research_artifact_contract"] = {
            "graded": "/workspace/policy.json",
            "analysis": "/workspace/research.py",
            "analysis_execution_verified": False,
            "verification": "fresh_public_runtime_during_scoring",
        }

    async def validate(self, trace: vf.Trace, runtime: vf.Runtime) -> bool:
        """Exercise the native artifact boundary with a model-free reference."""
        from .generator import resolve_episode
        from .verifier import reference_submission, score_submission

        episode = resolve_episode(self.data.id)
        reference = reference_submission(episode)
        expected = score_submission(episode, reference)["reward"]
        await runtime.write("/workspace/policy.json", json.dumps(reference, allow_nan=False).encode())
        await runtime.write("/workspace/research.py", files("financial_research").joinpath("reference.py").read_bytes())
        await self.finalize(trace, runtime)
        trace.state.artifacts = await vf.collect(
            runtime, self.data.artifacts, max_bytes=self.data.artifact_max_bytes)
        await vf.grade_submission(self, trace, runtime)
        return expected > 0 and math.isclose(trace.reward, expected, abs_tol=1e-12)

    async def stage_verifier(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        """Called only after artifacts enter a newly provisioned verifier box."""
        from .generator import resolve_episode

        _require_container(runtime)
        episode = resolve_episode(self.data.id)
        if (episode["id"] != self.data.id or episode["prompt"] != self.data.prompt
                or episode["public_files"] != self.data.public_files):
            raise ValueError("Task record differs from its registered episode")
        directory = f"/tmp/financial-verifier-{uuid4().hex}"
        package = files("financial_research")
        try:
            await runtime.write(f"{directory}/financial_research/__init__.py", b"")
            for name in ("generator.py", "verifier.py", "analytics.py"):
                await runtime.write(f"{directory}/financial_research/{name}",
                                    package.joinpath(name).read_bytes())
            await runtime.write(f"{directory}/episode.json",
                                json.dumps(episode, allow_nan=False).encode())
            await runtime.write(f"{directory}/score.py", _SCORE_PROGRAM.encode())
        except Exception:
            _outcome(trace, "infrastructure_error", "verifier_staging_failed", "staging", False)
            raise RuntimeError("verifier_staging_failed") from None
        trace.state.verifier_directory = directory
        trace.state.verifier_runtime = runtime.name
        trace.state.grade = None
        trace.state.reproduction = None
        trace.state.analysis_artifact_valid = False

    async def _grade(self, trace: vf.Trace, runtime: vf.Runtime) -> dict[str, Any]:
        from .reproduction import ReproductionInfrastructureError

        try:
            return await self._grade_checked(trace, runtime)
        except Exception as error:
            current = trace.info.get("financial_research_outcome", {})
            if current.get("status") != "infrastructure_error":
                verified = bool((trace.state.reproduction or {}).get("reproduction_verified"))
                _outcome(trace, "infrastructure_error", "scoring_error", "scoring", verified)
            reason = trace.info["financial_research_outcome"]["reason"]
            # Native traces persist formatted exception chains. Do not propagate
            # provider messages or their causes across that serialization boundary.
            if isinstance(error, ReproductionInfrastructureError):
                raise ReproductionInfrastructureError(reason) from None
            raise RuntimeError(reason) from None

    async def _grade_checked(self, trace: vf.Trace, runtime: vf.Runtime) -> dict[str, Any]:
        _require_container(runtime)
        if (trace.state.verifier_directory is None
                or trace.state.verifier_runtime != runtime.name):
            raise RuntimeError("Scoring requires the prepared isolated verifier")
        if trace.state.grade is None:
            reproduction = await self._reproduce(trace, runtime)
            if reproduction["status"] == "candidate_failure":
                trace.state.grade = _failure_grade(trace.state.analysis_artifact_valid)
                return trace.state.grade
            async with asyncio.timeout(self.config.scoring_seconds):
                result = await runtime.run(
                    ["python", "-I", f"{trace.state.verifier_directory}/score.py"], {})
            if result.exit_code != 0:
                # Do not mislabel a broken trusted grader as a candidate failure.
                _outcome(trace, "infrastructure_error", "trusted_grader_failed", "scoring", True)
                raise RuntimeError(f"Trusted verifier exited with status {result.exit_code}")
            grade = json.loads(result.stdout)
            if (set(grade) != {"reward", "metrics"}
                    or not isinstance(grade["metrics"], dict)):
                raise RuntimeError("Invalid trusted verifier result")
            values = [grade["reward"], *grade["metrics"].values()]
            if any(type(value) not in (int, float) or not math.isfinite(value)
                   for value in values):
                raise RuntimeError("Non-finite trusted verifier result")
            if not 0 <= grade["reward"] <= 1:
                raise RuntimeError("Trusted reward is outside [0, 1]")
            grade["metrics"]["reproduction_verified"] = 1.0
            grade["metrics"]["candidate_failure"] = float(grade["metrics"].get("schema_valid") == 0)
            if grade["metrics"]["candidate_failure"]:
                _outcome(trace, "candidate_failure", "invalid_policy_schema", "scoring", True)
            else:
                _outcome(trace, "passed", "scoring_completed", "scoring", True)
            trace.state.grade = grade
        return trace.state.grade

    async def _reproduce(self, trace: vf.Trace, runtime: vf.Runtime) -> dict[str, Any]:
        from .reproduction import ReproductionInfrastructureError, verify_reproduction

        if trace.state.reproduction is not None:
            return trace.state.reproduction
        try:
            analysis = _artifact_bytes(trace.state.artifacts, "/workspace/research.py", MAX_ANALYSIS_BYTES)
            try:
                if not analysis.decode("utf-8").strip():
                    raise CandidateArtifactError("invalid_analysis_artifact")
            except UnicodeError as error:
                raise CandidateArtifactError("invalid_analysis_artifact") from error
            trace.state.analysis_artifact_valid = True
            policy_bytes = _artifact_bytes(trace.state.artifacts, "/workspace/policy.json", MAX_POLICY_BYTES)
            submitted = _parse_policy(policy_bytes)
        except CandidateArtifactError as error:
            result = {"status": "candidate_failure", "reason": str(error), "reproduction_verified": False}
            trace.state.reproduction = result
            _outcome(trace, "candidate_failure", str(error), "artifacts", False)
            return result
        try:
            result = await verify_reproduction(
                self.data.public_files, analysis, submitted,
                runtime_config=runtime.config,
                timeout_seconds=30,
            )
        except ReproductionInfrastructureError as error:
            _outcome(trace, "infrastructure_error", str(error), "reproduction", False)
            raise
        if (result.get("status") not in ("passed", "candidate_failure")
                or not isinstance(result.get("reason"), str)
                or type(result.get("reproduction_verified")) is not bool
                or (result["status"] == "passed") != result["reproduction_verified"]):
            _outcome(trace, "infrastructure_error", "invalid_reproduction_result", "reproduction", False)
            raise RuntimeError("Invalid trusted reproduction result")
        trace.state.reproduction = result
        _outcome(trace, result["status"], result["reason"], "reproduction", result["reproduction_verified"])
        trace.info["research_artifact_contract"] = {
            "graded": "/workspace/policy.json", "analysis": "/workspace/research.py",
            "analysis_execution_verified": result["reproduction_verified"],
            "verification": "fresh_public_runtime_during_scoring",
        }
        return result

    @vf.metric
    async def research_metrics(self, trace: vf.Trace, runtime: vf.Runtime) -> dict[str, float]:
        return (await self._grade(trace, runtime))["metrics"]

    @vf.reward
    async def research_reward(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        return float((await self._grade(trace, runtime))["reward"])


class CandidateArtifactError(ValueError):
    """A submitted artifact violates the public submission contract."""


def _artifact_bytes(artifacts: dict[str, bytes | None], path: str, limit: int) -> bytes:
    """Read one regular member from native archive bytes without extraction."""
    archive = artifacts.get(path)
    label = "analysis" if path.endswith("research.py") else "policy"
    if archive is None:
        raise CandidateArtifactError(f"missing_{label}_artifact")
    if not isinstance(archive, bytes) or len(archive) > 4 * 1024 * 1024:
        raise CandidateArtifactError(f"invalid_{label}_archive")
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
            members = bundle.getmembers()
            if len(members) != 1:
                raise CandidateArtifactError(f"invalid_{label}_archive")
            member = members[0]
            if (member.name != path.lstrip("/") or not member.isfile()
                    or member.issparse() or not 0 < member.size <= limit):
                raise CandidateArtifactError(f"invalid_{label}_artifact")
            with bundle.extractfile(member) as stream:
                payload = stream.read(limit + 1)
            if len(payload) != member.size:
                raise CandidateArtifactError(f"invalid_{label}_artifact")
            return payload
    except (tarfile.TarError, OSError, EOFError) as error:
        raise CandidateArtifactError(f"invalid_{label}_archive") from error


def _parse_policy(payload: bytes) -> dict[str, Any]:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError("non-finite number")

    try:
        policy = json.loads(payload, object_pairs_hook=unique_object, parse_constant=reject_constant)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise CandidateArtifactError("invalid_policy_json") from error
    if not isinstance(policy, dict):
        raise CandidateArtifactError("invalid_policy_json")
    return policy


def _failure_grade(analysis_valid: bool = False) -> dict[str, Any]:
    return {"reward": 0.0, "metrics": {
        "schema_valid": 0.0, "audit_accuracy": 0.0, "trial_accuracy": 0.0,
        "decision_correct": 0.0, "economic_quality": 0.0,
        "analysis_artifact_valid": float(analysis_valid), "reproduction_verified": 0.0,
        "candidate_failure": 1.0,
    }}


def _outcome(trace: vf.Trace, status: str, reason: str, phase: str, verified: bool) -> None:
    result = {"status": status, "reason": reason, "phase": phase,
              "reproduction_verified": verified}
    trace.info["financial_research_outcome"] = result
    # Native isolated grading preserves this key even when a hook raises.
    trace.info.setdefault("scoring_diagnostics", {})["financial_research_outcome"] = dict(result)


def _require_container(runtime: vf.Runtime) -> None:
    if runtime.type not in ("docker", "prime"):
        raise ValueError("Use a Docker or Prime container runtime")
    if getattr(runtime.config, "mounts", {}):
        raise ValueError("Research solver and verifier runtimes must not mount host data")
    if "FINANCIAL_RESEARCH_KEY" in getattr(runtime, "env", {}):
        raise ValueError("The research generation key must remain on the trusted worker")


class ResearchTasksetConfig(vf.TasksetConfig):
    num_tasks: int = Field(default=12, gt=0, le=128, strict=True)
    seed_start: int | None = Field(default=None, ge=0, strict=True)
    split: Literal["train", "dev", "test"] = "train"
    family: Literal["signal", "null", "cost_trap", "revision_trap"] | None = None
    task: ResearchTaskConfig = Field(default_factory=ResearchTaskConfig)

    @field_validator("num_tasks", "seed_start", mode="before")
    @classmethod
    def parse_cli_integer(cls, value):
        # The native CLI supplies option values as strings, even for strict ints.
        if isinstance(value, str) and value.isascii() and value.isdecimal():
            return int(value)
        return value

    @model_validator(mode="after")
    def validate_registered_range(self):
        lower, upper = {"train": (0, 128), "dev": (128, 160), "test": (160, 192)}[self.split]
        start = lower if self.seed_start is None else self.seed_start
        if start < lower or start + self.num_tasks > upper:
            raise ValueError("Requested tasks exceed the registered split seed range")
        self.seed_start = start
        return self


class FinancialResearchTaskset(vf.Taskset[FinancialResearchTask, ResearchTasksetConfig]):
    def load(self):
        from .generator import build_episode

        for seed in range(self.config.seed_start, self.config.seed_start + self.config.num_tasks):
            episode = build_episode(seed, split=self.config.split, family=self.config.family)
            data = ResearchData(
                id=episode["id"], prompt=episode["prompt"],
                public_files=episode["public_files"],
                image=self.config.task.image, workdir="/workspace",
                network=vf.NetworkPolicyConfig(allow=[]),
                scoring=vf.ScoringConfig(mode="isolated", network=vf.NetworkPolicyConfig(allow=[])),
                artifacts=[vf.Artifact(source="/workspace/policy.json", required=False),
                           vf.Artifact(source="/workspace/research.py", required=False)],
                artifact_max_bytes=4 * 1024 * 1024,
                timeout=vf.TaskTimeout(setup=120, agent=600, finalize=30,
                                       scoring=self.config.task.scoring_seconds),
                resources=vf.TaskResources(cpu=1, memory=1),
            )
            yield FinancialResearchTask(data, self.config.task)
