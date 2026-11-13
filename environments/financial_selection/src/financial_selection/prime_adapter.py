"""Native Verifiers 0.4.1 taskset for public-only program reproduction.

Candidate code executes exclusively in separate public-data containers. Trusted
worker hooks regenerate keyed packets and score declarative bytes; no key,
population, fresh packet, or per-packet answer is placed in TaskData or traces.
This local adapter is an evaluation contract, not evidence of policy training.
"""
from __future__ import annotations

import asyncio
import base64
from importlib.resources import files
import io
import json
import math
import re
import tarfile
from typing import Any, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator
import verifiers.v1 as vf
from verifiers.v1.configs.retries import RetryConfig

from .contract import PUBLIC_FILES, parse_public
from . import registry, replay


MAX_POLICY_BYTES = 2 * 1024 * 1024
MAX_SOURCE_BYTES = 1024 * 1024
INFO_KEY = "financial_selection_outcome"
__all__ = ["FinancialSelectionTaskset"]

# Read bytes, never import or execute the submitted program. Opening the directory
# and files without following links closes both root- and leaf-symlink cases.
_CAPTURE = r'''
import base64,json,os,stat,sys
result={}
try:
    directory=os.open(sys.argv[1],os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
except OSError:
    directory=None
for name,limit in (("research.py",1048576),("policy.json",2097152)):
    result[name]=None
    if directory is None:
        continue
    fd=None
    try:
        fd=os.open(name,os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW,dir_fd=directory)
        metadata=os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode) or not 0<metadata.st_size<=limit:
            continue
        payload=b""
        while len(payload)<=limit:
            chunk=os.read(fd,min(65536,limit+1-len(payload)))
            if not chunk:
                break
            payload+=chunk
        if 0<len(payload)<=limit:
            result[name]=base64.b64encode(payload).decode("ascii")
    except OSError:
        pass
    finally:
        if fd is not None:
            os.close(fd)
if directory is not None:
    os.close(directory)
print(json.dumps(result,separators=(",",":")))
'''


def _regular_archive(path, payload):
    target = io.BytesIO()
    with tarfile.open(fileobj=target, mode="w") as archive:
        member = tarfile.TarInfo(path.lstrip("/"))
        member.size = len(payload)
        member.mode = 0o600
        archive.addfile(member, io.BytesIO(payload))
    return target.getvalue()


async def _capture_artifacts(runtime):
    completed = await runtime.run(["python", "-I", "-c", _CAPTURE, "/workspace"], {})
    if completed.exit_code or len(completed.stdout.encode("utf-8")) > 5 * 1024 * 1024:
        raise RuntimeError("artifact_capture_unavailable")
    captured = json.loads(completed.stdout)
    if type(captured) is not dict or set(captured) != {"research.py", "policy.json"}:
        raise RuntimeError("artifact_capture_unavailable")
    artifacts = {}
    for name, limit in (("research.py", MAX_SOURCE_BYTES), ("policy.json", MAX_POLICY_BYTES)):
        encoded = captured[name]
        path = "/workspace/" + name
        if encoded is None:
            artifacts[path] = None
            continue
        if type(encoded) is not str:
            raise RuntimeError("artifact_capture_unavailable")
        payload = base64.b64decode(encoded, validate=True)
        if not 0 < len(payload) <= limit:
            raise RuntimeError("artifact_capture_unavailable")
        artifacts[path] = _regular_archive(path, payload)
    return artifacts
PROMPT = """Complete the synthetic selection-adjusted research and funded-allocation
task using only the five UTF-8 files in /workspace: contract.json,
deployment.json, universe.csv, feature_releases.csv, observed_labels.csv.
Read contract.json for all financial equations, information cutoffs, support
rules, finite allocation constraints and tie breaks. Time is an integer tick;
money is cents; markout is basis points; rates apply to one holding period.
This is prescribed public-data calculation credit, not missing-outcome recovery,
post-selection confidence coverage or realized investment performance.

Create /workspace/research.py, a self-contained Python 3 standard-library
program that reads those five files from its working directory and writes
/workspace/policy.json. The program must solve the supplied packet again when
run on different generated inputs. Network access is disabled. There are no
installed task, generator, evaluator, or reference modules in execution boxes.
The final conversational response is not graded. Each artifact must be a regular
nonempty UTF-8 file: research.py at most 1 MiB, policy.json at most 2 MiB.

policy.json is strict JSON with exactly these top-level keys:
schema, input_sha256, audit, cells, baskets, basket_covariance_cents2,
allocation, ledger. schema is "selection-policy-v1". No duplicate JSON keys,
floating-point literals, NaN, infinity or boolean quantities are accepted.
Rationals are exact reduced [integer numerator, positive integer denominator]
pairs; integers have at most 4096 decimal digits. Nulls follow the support rules.
input_sha256 hashes the exact bytes of all five inputs: in lexicographic filename
order append UTF-8 filename, one NUL byte, unsigned 8-byte big-endian payload
length, then exact UTF-8 payload; take SHA-256 of the concatenation.

Output rows contain exactly the following keys:
* audit: event_id, population_member (boolean), cell_id, feature_release_id,
  label_release_id (IDs or null). Rows follow sorted audit_event_ids. Unmatured
  events have three null IDs; matured unarchived events have null label IDs.
* cells: cell_id, population_count, observed_count (integer counts), status
  ("unsupported", "insufficient_sample", "eligible"), mean_bp, variance_bp2,
  effective_sample_size (each rational or null). Rows follow sorted cell IDs.
* baskets: basket_id, eligible (boolean), mean_bp (rational or null), sorted by ID.
* basket_covariance_cents2: full square matrix in sorted basket order, each entry
  rational or null. Null if either basket mean is unsupported; supported but
  sample-gated baskets retain numerical covariance. Include both triangles.
* allocation: one exact integer lot count per basket in sorted order.
* ledger integer fields: invested_principal_cents, entry_cost_cents,
  exit_cost_cents, pretrade_cash_cents, penalty_microcents, objective_microcents.
  Ledger rational fields: interest_cents, estimated_gross_cents,
  estimated_terminal_cents, cash_benchmark_cents, estimated_excess_cents,
  sampling_variance_cents2. No other ledger keys are allowed.

The exact public verifier awards 0.20 audit, 0.35 statistical calculations,
0.15 submitted-allocation cash ledger, and 0.30 canonical optimal allocation.
Audit credit averages the four non-ID fields per requested row (empty audit
earns that component). Statistical credit averages six non-ID fields per cell,
two per basket and every covariance entry, including both triangles. Ledger
credit averages its twelve fields for the submitted feasible allocation.
Schema/identity errors or infeasible lots earn zero; wrong well-formed values
lose their corresponding field credit. Private outcomes do not affect credit.

The exact submitted source is run once in a fresh container on this packet.
Its parsed valid policy must exactly equal the submitted policy; failure zeros
the aggregate. The same source then runs once on each of two distinct fresh
packets, in separate fresh public-only containers. Final reward is the arithmetic
mean of these three public calculation scores. Fresh-input execution/output
failures score zero for that packet. Each program has 30 seconds wall time,
256 MiB address space and bounded process/file resources. Infrastructure errors
invalidate the evaluation rather than becoming financial zeros. This checks
this source on these three inputs; it does not certify unrestricted generalization.
"""


def _immutable_image(value):
    return isinstance(value, str) and re.fullmatch(r"(?:sha256:|[A-Za-z0-9._:/-]+@sha256:)[0-9a-f]{64}", value) is not None


class SelectionData(vf.TaskData):
    model_config = ConfigDict(frozen=True, extra="forbid")
    public_files: dict[str, str]

    @field_validator("public_files")
    @classmethod
    def validate_public(cls, value):
        parse_public(value)
        return value


class SelectionState(vf.State):
    solver_runtime: str | None = None
    verifier_runtime: str | None = None
    grade: dict[str, Any] | None = None


class SelectionTaskConfig(vf.TaskConfig):
    image: str | None = None
    scoring_seconds: float = Field(default=240, gt=0, le=600, allow_inf_nan=False)

    @field_validator("image")
    @classmethod
    def validate_image(cls, value):
        if value is not None and not _immutable_image(value):
            raise ValueError("An immutable Docker image identity is required")
        return value

    @model_validator(mode="after")
    def fixed_scoring_contract(self):
        if self.judges or self.stops or self.metrics or self.rewards:
            raise ValueError("This experimental scoring contract does not support callback overrides")
        return self


def _outcome(trace, status, reason, reproduced=False):
    value = {"status": status, "reason": reason, "reproduction_verified": reproduced}
    trace.info[INFO_KEY] = value
    trace.info.setdefault("scoring_diagnostics", {})[INFO_KEY] = dict(value)


def evaluation_admission(trace):
    """Fail-closed denominator gate; native zero alone is never admission evidence."""
    if (trace.errors or trace.is_timeout or trace.dropped
            or getattr(trace, "is_format_violation", False)):
        return {"performance_usable": False, "status": "infrastructure_or_native_failure"}
    outcome = trace.info.get(INFO_KEY, {})
    status = outcome.get("status")
    if status not in {"scored", "candidate_failure"}:
        return {"performance_usable": False, "status": "unresolved"}
    reward = trace.rewards.get("selection_reward")
    names = {"candidate_failure", "reproduction_verified", "packets_scored",
             "exact_success_fraction", "all_packets_correct", "current_exact_success",
             "fresh_exact_success_fraction"}
    if (set(trace.rewards) != {"selection_reward"}
            or reward is None or reward.weight != 1 or not math.isfinite(reward.score)
            or not 0 <= reward.score <= 1 or not names <= set(trace.metrics)
            or any(type(trace.metrics[n]) not in (int, float) or not math.isfinite(trace.metrics[n]) for n in names)):
        return {"performance_usable": False, "status": "incomplete_scoring"}
    m = trace.metrics
    cached = getattr(trace.state, "grade", None)
    if (trace.reward != reward.score
            or trace.info.get("scoring_diagnostics", {}).get(INFO_KEY) != outcome
            or (cached is not None and cached != {"reward": reward.score, "metrics": {n: m[n] for n in names}})):
        return {"performance_usable": False, "status": "inconsistent_scoring"}
    if status == "candidate_failure":
        consistent = (reward.score == 0 and outcome.get("reproduction_verified") is False
                      and m["candidate_failure"] == 1 and all(m[n] == 0 for n in names - {"candidate_failure"}))
    else:
        consistent = (outcome.get("reproduction_verified") is True and m["candidate_failure"] == 0
                      and m["reproduction_verified"] == 1 and m["packets_scored"] == 3
                      and m["current_exact_success"] in (0, 1) and m["fresh_exact_success_fraction"] in (0, .5, 1)
                      and m["exact_success_fraction"] == (m["current_exact_success"] + 2*m["fresh_exact_success_fraction"])/3
                      and m["all_packets_correct"] == float(m["exact_success_fraction"] == 1)
                      and reward.score >= m["exact_success_fraction"]
                      and (reward.score == 1) == (m["all_packets_correct"] == 1))
    return {"performance_usable": bool(consistent), "status": status if consistent else "inconsistent_scoring"}


class FinancialSelectionTask(vf.Task[SelectionData, SelectionState, SelectionTaskConfig]):
    NEEDS_CONTAINER = True

    def _require_runtime(self, runtime):
        config = runtime.config
        if (runtime.type != "docker" or not _immutable_image(self.config.image)
                or config.image != self.config.image or self.data.image != self.config.image
                or getattr(config, "mounts", {}) or getattr(config, "checkpoint", None)
                or getattr(config, "allow", ["*"]) != []
                or getattr(runtime, "env", {})):
            raise RuntimeError("unsafe_selection_runtime")

    def _resolve_bound(self):
        packets = registry.resolve(self.data.id)
        if (self.data.prompt != PROMPT or self.data.public_files != packets.public_files
                or set(self.data.public_files) != set(PUBLIC_FILES)):
            raise registry.RegistryError("task_public_binding_mismatch")
        return packets

    def grading_task(self, runtime, *, scoring):
        self._require_runtime(runtime)
        if (scoring.mode != "isolated" or scoring.runtime is not None or scoring.env
                or scoring.retries.max_retries != 0
                or scoring.skip_on_timeout or scoring.skip_on_truncated
                or scoring.network is None or scoring.network.allow != []):
            raise RuntimeError("unsafe_selection_scoring_policy")
        return self

    async def setup(self, runtime, trace=None):
        try:
            self._require_runtime(runtime)
            self._resolve_bound()
            for name, content in self.data.public_files.items():
                await runtime.write("/workspace/" + name, content.encode("utf-8"))
        except Exception:
            if trace is not None:
                _outcome(trace, "infrastructure_error", "public_setup_failed")
            raise RuntimeError("public_setup_failed") from None

    async def finalize(self, trace, runtime):
        self._require_runtime(runtime)
        trace.state.solver_runtime = runtime.name
        try:
            trace.state.artifacts = await _capture_artifacts(runtime)
        except asyncio.CancelledError:
            _outcome(trace, "infrastructure_error", "artifact_capture_cancelled")
            raise
        except Exception:
            _outcome(trace, "infrastructure_error", "artifact_capture_failed")
            raise RuntimeError("artifact_capture_failed") from None
        trace.info["selection_artifact_contract"] = {
            "analysis": "/workspace/research.py", "graded": "/workspace/policy.json",
            "reproduction_verified": False, "fresh_inputs": 2,
        }

    async def stage_verifier(self, trace, runtime):
        try:
            self._require_runtime(runtime)
            if trace.state.solver_runtime is None or trace.state.solver_runtime == runtime.name:
                raise RuntimeError("isolated_verifier_required")
            self._resolve_bound()
        except Exception:
            _outcome(trace, "infrastructure_error", "verifier_binding_failed")
            raise RuntimeError("verifier_binding_failed") from None
        trace.state.verifier_runtime = runtime.name
        trace.state.grade = None

    async def _grade(self, trace, runtime):
        # Native metric and reward hooks may run concurrently. Cache only the
        # filtered aggregate, and execute the three-container check exactly once.
        if not hasattr(self, "_scoring_lock"):
            self._scoring_lock = asyncio.Lock()
        async with self._scoring_lock:
            try:
                self._require_runtime(runtime)
                if trace.state.verifier_runtime != runtime.name or runtime.name == trace.state.solver_runtime:
                    raise RuntimeError("isolated_verifier_required")
                if trace.state.grade is not None:
                    return trace.state.grade
                packets = self._resolve_bound()
                from financial_research.prime_adapter import CandidateArtifactError, _artifact_bytes
                try:
                    source = _artifact_bytes(trace.state.artifacts, "/workspace/research.py", MAX_SOURCE_BYTES)
                    policy = _artifact_bytes(trace.state.artifacts, "/workspace/policy.json", MAX_POLICY_BYTES)
                    if not replay.source_valid(source):
                        raise CandidateArtifactError("invalid_analysis_artifact")
                except CandidateArtifactError:
                    grade = {"reward": 0.0, "metrics": {
                        "candidate_failure": 1.0, "reproduction_verified": 0.0, "packets_scored": 0.0,
                        "exact_success_fraction": 0.0, "all_packets_correct": 0.0,
                        "current_exact_success": 0.0, "fresh_exact_success_fraction": 0.0,
                    }}
                    trace.state.grade = grade
                    _outcome(trace, "candidate_failure", "invalid_submission_artifacts")
                    return grade
                async with asyncio.timeout(self.config.scoring_seconds):
                    checked = await replay.verify_program(packets.public_files, source, policy, packets.fresh_packets,
                                                          runtime_config=runtime.config, timeout_seconds=30)
                reward = checked.get("reward")
                reproduced = checked.get("reproduction_verified")
                if (type(reward) not in (int, float) or not math.isfinite(reward) or not 0 <= reward <= 1
                        or type(reproduced) is not bool or checked.get("status") not in ("scored", "candidate_failure")):
                    raise RuntimeError("invalid_replay_result")
                if checked["status"] == "scored":
                    packet_rewards = checked.get("packet_rewards")
                    if (not reproduced or not isinstance(packet_rewards, list) or len(packet_rewards) != 3
                            or any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1 for x in packet_rewards)
                            or not math.isclose(reward, sum(packet_rewards) / 3, abs_tol=1e-12)):
                        raise RuntimeError("invalid_replay_result")
                    status, reason = "scored", "three_packet_program_checked"
                else:
                    if reward != 0 or reproduced:
                        raise RuntimeError("invalid_replay_result")
                    status, reason = "candidate_failure", "program_reproduction_failed"
                grade = {"reward": float(reward), "metrics": {
                    "candidate_failure": float(status == "candidate_failure"),
                    "reproduction_verified": float(reproduced), "packets_scored": 3.0 if reproduced else 0.0,
                    "exact_success_fraction": sum(x == 1 for x in packet_rewards) / 3 if reproduced else 0.0,
                    "all_packets_correct": float(all(x == 1 for x in packet_rewards)) if reproduced else 0.0,
                    "current_exact_success": float(packet_rewards[0] == 1) if reproduced else 0.0,
                    "fresh_exact_success_fraction": sum(x == 1 for x in packet_rewards[1:]) / 2 if reproduced else 0.0,
                }}
                trace.state.grade = grade
                trace.info["selection_artifact_contract"] = {
                    "analysis": "/workspace/research.py", "graded": "/workspace/policy.json",
                    "reproduction_verified": reproduced, "fresh_inputs": 2,
                }
                _outcome(trace, status, reason, reproduced)
                return grade
            except asyncio.CancelledError:
                _outcome(trace, "infrastructure_error", "scoring_cancelled")
                raise
            except Exception:
                _outcome(trace, "infrastructure_error", "scoring_service_failed")
                raise replay.ReplayInfrastructureError("scoring_service_failed") from None

    @vf.metric
    async def selection_metrics(self, trace: vf.Trace, runtime: vf.Runtime) -> dict[str, float]:
        return (await self._grade(trace, runtime))["metrics"]

    @vf.reward
    async def selection_reward(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        return (await self._grade(trace, runtime))["reward"]

    async def validate(self, trace: vf.Trace, runtime: vf.Runtime) -> bool:
        """Model-free native collection, isolated grading and three-input replay."""
        from .reference import solve
        self._require_runtime(runtime)
        self._resolve_bound()
        await runtime.write("/workspace/research.py", files("financial_selection").joinpath("reference.py").read_bytes())
        await runtime.write("/workspace/policy.json", json.dumps(solve(self.data.public_files), allow_nan=False).encode())
        await self.finalize(trace, runtime)
        await vf.grade_submission(self, trace, runtime)
        return math.isclose(trace.reward, 1.0, abs_tol=1e-12) and trace.metrics.get("reproduction_verified") == 1.0


class SelectionTasksetConfig(vf.TasksetConfig):
    num_tasks: int = Field(default=4, ge=1, le=64, strict=True)
    index_start: int = Field(default=0, ge=0, le=63, strict=True)
    split: Literal["train", "dev", "final"] = "dev"
    condition: Literal["mixed", "positive", "negative", "unsupported", "sparse"] = "mixed"
    task: SelectionTaskConfig = Field(default_factory=SelectionTaskConfig)

    @field_validator("num_tasks", "index_start", mode="before")
    @classmethod
    def parse_cli_integer(cls, value):
        if isinstance(value, str) and value.isascii() and value.isdecimal():
            return int(value)
        return value

    @model_validator(mode="after")
    def bounded_selection(self):
        if self.index_start + self.num_tasks > registry.CURRENT_INDEX_COUNT:
            raise ValueError("Selection exceeds the bounded current-packet registry")
        return self


class FinancialSelectionTaskset(vf.Taskset[FinancialSelectionTask, SelectionTasksetConfig]):
    def load(self):
        if not _immutable_image(self.config.task.image):
            raise ValueError("Configure an immutable local Docker image before loading tasks")
        episodes = [registry.public_episode(self.config.split, index, self.config.condition)
                    for index in range(self.config.index_start, self.config.index_start + self.config.num_tasks)]
        for identity, public in sorted(episodes):
            data = SelectionData(
                id=identity, prompt=PROMPT, public_files=public,
                image=self.config.task.image, workdir="/workspace", network=vf.NetworkPolicyConfig(allow=[]),
                scoring=vf.ScoringConfig(mode="isolated", network=vf.NetworkPolicyConfig(allow=[]),
                                        retries=RetryConfig(max_retries=0)),
                artifacts=[vf.Artifact(source="/workspace/policy.json", required=False),
                           vf.Artifact(source="/workspace/research.py", required=False)],
                artifact_max_bytes=4 * 1024 * 1024,
                timeout=vf.TaskTimeout(setup=120, agent=600, finalize=30, scoring=self.config.task.scoring_seconds),
                resources=vf.TaskResources(cpu=1, memory=1),
            )
            yield FinancialSelectionTask(data, self.config.task)
