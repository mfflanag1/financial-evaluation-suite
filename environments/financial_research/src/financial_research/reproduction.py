"""Rerun submitted analysis in a new public-only runtime; never in the private grader."""

import asyncio
import base64
import json
import math

PUBLIC_FILES = frozenset({"prices.csv", "features.csv", "securities.csv", "contract.json"})
MAX_POLICY_BYTES = 2 * 1024 * 1024
MAX_SOURCE_BYTES = 1024 * 1024
METRICS = ("mean_net", "sharpe", "mean_turnover", "screening_score")


class ReproductionInfrastructureError(RuntimeError):
    """Execution service failure, distinct from invalid candidate submissions."""


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("nonfinite JSON value")


def parse_policy(payload):
    if type(payload) is not bytes or len(payload) > MAX_POLICY_BYTES:
        raise ValueError("invalid policy size")
    return json.loads(payload, object_pairs_hook=_unique, parse_constant=_reject_constant)


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def normalize_policy(policy, contract):
    """Validate against public schema; order is immaterial, decisions are exact."""
    if type(policy) is not dict or set(policy) != {"audit", "trials", "decision", "selected"}:
        raise ValueError("invalid policy schema")
    requested = {(r["entity_id"], r["day"], r["field"]) for r in contract["audit_requests"]}
    candidates = {r["id"] for r in contract["candidates"]}
    audits, trials = {}, {}
    if type(policy["audit"]) is not list or type(policy["trials"]) is not list:
        raise ValueError("invalid policy rows")
    if len(policy["audit"]) != len(requested) or len(policy["trials"]) != len(candidates):
        raise ValueError("incomplete policy rows")
    for row in policy["audit"]:
        if type(row) is not dict or set(row) != {"entity_id", "day", "field", "value"}:
            raise ValueError("invalid audit row")
        if type(row["entity_id"]) is not str or type(row["field"]) is not str or type(row["day"]) is not int:
            raise ValueError("invalid audit identity")
        key = (row["entity_id"], row["day"], row["field"])
        if key not in requested or key in audits or (row["value"] is not None and not _finite(row["value"])):
            raise ValueError("invalid audit value")
        audits[key] = row["value"]
    for row in policy["trials"]:
        if type(row) is not dict or set(row) != {"id", *METRICS} or type(row["id"]) is not str:
            raise ValueError("invalid trial row")
        if row["id"] not in candidates or row["id"] in trials or not all(_finite(row[m]) for m in METRICS):
            raise ValueError("invalid trial value")
        trials[row["id"]] = {metric: row[metric] for metric in METRICS}
    selected = policy["selected"]
    if selected is not None and (type(selected) is not str or selected not in candidates):
        raise ValueError("invalid selection")
    if policy["decision"] not in ("deploy", "abstain") or (policy["decision"] == "abstain") != (selected is None):
        raise ValueError("inconsistent decision")
    return audits, trials, policy["decision"], selected


def policies_match(left, right, contract):
    a, t, decision, selected = normalize_policy(left, contract)
    b, u, other_decision, other_selected = normalize_policy(right, contract)
    if (decision, selected) != (other_decision, other_selected):
        return False
    for key, value in a.items():
        if value is None or b[key] is None:
            if value is not b[key]:
                return False
        elif abs(value - b[key]) > 1e-8:
            return False
    return all(abs(row[metric] - u[key][metric]) <= (1e-6 if metric in ("sharpe", "mean_turnover") else 1e-8)
               for key, row in t.items() for metric in METRICS)


def outcome(status, reason):
    return {"status": status, "reason": reason, "reproduction_verified": status == "passed"}


def fresh_config(base):
    from verifiers.v1.runtimes import DockerConfig, PrimeConfig
    if not isinstance(base, (DockerConfig, PrimeConfig)):
        raise ReproductionInfrastructureError("unsupported_reproduction_runtime")
    if base.mounts or getattr(base, "checkpoint", None):
        raise ReproductionInfrastructureError("unsafe_reproduction_configuration")
    # Do not inherit environment, labels, mounts, checkpoints, or relaxed egress.
    return type(base)(image=base.image, workdir="/workspace", cpu=1, memory=1, allow=[])


# The root-owned controller is staged only into the new public runtime. Candidate
# code executes as an unprivileged user and cannot replace it or its result file.
_CONTROLLER = r'''
import base64, ctypes, errno, json, math, os, resource, signal, stat, subprocess, sys, time
from pathlib import Path

root = Path(__file__).parent
limit = float(sys.argv[1])
os.umask(0o077)
if os.geteuid() != 0:
    raise RuntimeError("controller requires root to drop candidate privileges")
os.chown("/workspace", 65534, 65534)
os.chmod("/workspace", 0o700)

def restrict():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (4*1024*1024, 4*1024*1024))
    resource.setrlimit(resource.RLIMIT_AS, (256*1024*1024, 256*1024*1024))
    resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    cpu = max(1, math.ceil(limit))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    if ctypes.CDLL(None, use_errno=True).prctl(38, 1, 0, 0, 0) != 0:
        raise OSError("cannot restrict privileges")
    os.setgroups([])
    os.setgid(65534)
    os.setuid(65534)

def remove_children():
    # A candidate may start detached children; stop every process of its sole UID
    # in this dedicated runtime before opening the output file.
    for _ in range(5):
        found = False
        for entry in Path("/proc").iterdir():
            if not entry.name.isdecimal():
                continue
            try:
                data = (entry / "status").read_text()
                uid = next(line for line in data.splitlines() if line.startswith("Uid:")).split()[1]
                state = next(line for line in data.splitlines() if line.startswith("State:")).split()[1]
                if uid == "65534" and state != "Z":
                    found = True
                    os.kill(int(entry.name), signal.SIGKILL)
            except (FileNotFoundError, ProcessLookupError):
                pass
        if not found:
            return
        time.sleep(0.02)
    raise RuntimeError("candidate cleanup could not be confirmed")

result = {}
try:
    with open(os.devnull, "wb") as sink:
        process = subprocess.Popen([sys.executable, "-I", "-B", "/workspace/research.py"],
            cwd="/workspace", env={"PATH":"/usr/local/bin:/usr/bin:/bin","LANG":"C.UTF-8"},
            stdin=subprocess.DEVNULL, stdout=sink, stderr=sink, preexec_fn=restrict,
            start_new_session=True)
        try:
            code = process.wait(timeout=limit)
            if code != 0:
                result = {"reason":"analysis_exit"}
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
            result = {"reason":"analysis_timeout"}
finally:
    remove_children()

if not result:
    try:
        metadata = os.lstat("/workspace/policy.json")
        if stat.S_ISLNK(metadata.st_mode):
            result = {"reason":"output_missing_or_unsafe"}
        elif not stat.S_ISREG(metadata.st_mode):
            result = {"reason":"output_not_regular"}
        else:
            fd = os.open("/workspace/policy.json", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, "rb") as stream:
                metadata = os.fstat(stream.fileno())
                if not stat.S_ISREG(metadata.st_mode):
                    result = {"reason":"output_not_regular"}
                elif metadata.st_size > 2097152:
                    result = {"reason":"output_too_large"}
                else:
                    payload = stream.read(2097153)
                    result = ({"reason":"output_too_large"} if len(payload)>2097152 else
                              {"reason":"output_created","policy":base64.b64encode(payload).decode("ascii")})
    except OSError as error:
        if error.errno in (errno.ENOENT, errno.ELOOP, errno.EACCES, errno.ENOTDIR):
            result = {"reason":"output_missing_or_unsafe"}
        else:
            raise
(root / "result.json").write_text(json.dumps(result, allow_nan=False))
'''


async def verify_reproduction(public_files, analysis_bytes, submitted_policy, *, runtime_config, timeout_seconds=30):
    """Return candidate outcomes; raise sanitized infrastructure errors separately."""
    from verifiers.v1.runtimes import provision_runtime
    if set(public_files) != PUBLIC_FILES or not all(type(v) is str for v in public_files.values()):
        raise ReproductionInfrastructureError("invalid_public_inputs")
    if not _finite(timeout_seconds) or not 0 < timeout_seconds <= 120:
        raise ReproductionInfrastructureError("invalid_reproduction_timeout")
    if type(analysis_bytes) is not bytes or not 0 < len(analysis_bytes) <= MAX_SOURCE_BYTES:
        return outcome("candidate_failure", "invalid_analysis_artifact")
    try:
        if not analysis_bytes.decode("utf-8").strip():
            return outcome("candidate_failure", "invalid_analysis_artifact")
    except UnicodeError:
        return outcome("candidate_failure", "invalid_analysis_artifact")
    try:
        contract = json.loads(public_files["contract.json"])
    except (ValueError, UnicodeError, RecursionError):
        raise ReproductionInfrastructureError("invalid_public_contract") from None
    try:
        normalize_policy(submitted_policy, contract)
    except (ValueError, TypeError, OverflowError, RecursionError):
        return outcome("candidate_failure", "invalid_submitted_policy")
    config = fresh_config(runtime_config)
    try:
        async with provision_runtime(config, env={}) as runtime:
            await runtime.backend.prepare_setup()
            await runtime.write("/opt/financial-reproduction/controller.py", _CONTROLLER.encode())
            setup = await runtime.run(["chmod", "700", "/opt/financial-reproduction"], {})
            if setup.exit_code:
                raise ReproductionInfrastructureError("reproduction_setup_failed")
            for name, content in public_files.items():
                await runtime.write(f"/workspace/{name}", content.encode())
            await runtime.write("/workspace/research.py", analysis_bytes)
            await runtime.backend.prepare_execution([])
            # The inner controller accounts candidate deadlines. A transport or
            # controller deadline is a service failure, not a candidate timeout.
            async with asyncio.timeout(timeout_seconds + 15):
                completed = await runtime.run(["python", "-I", "/opt/financial-reproduction/controller.py",
                                               str(timeout_seconds)], {})
            if completed.exit_code:
                raise ReproductionInfrastructureError("reproduction_controller_failed")
            raw = await runtime.read("/opt/financial-reproduction/result.json", max_bytes=3*1024*1024)
            captured = json.loads(raw)
            if captured.get("reason") != "output_created":
                allowed = {"analysis_exit", "analysis_timeout", "output_not_regular", "output_too_large", "output_missing_or_unsafe"}
                if captured.get("reason") not in allowed:
                    raise ReproductionInfrastructureError("invalid_controller_result")
                return outcome("candidate_failure", captured["reason"])
            try:
                reproduced = parse_policy(base64.b64decode(captured["policy"], validate=True))
                match = policies_match(submitted_policy, reproduced, contract)
            except (ValueError, TypeError, KeyError, UnicodeError, OverflowError, RecursionError):
                return outcome("candidate_failure", "invalid_reproduced_policy")
            return outcome("passed", "policy_reproduced") if match else outcome("candidate_failure", "policy_mismatch")
    except ReproductionInfrastructureError:
        raise
    except Exception:
        # Never persist provider messages or runtime configuration, which may hold
        # credentials. Cancellation still propagates to the owning lifecycle.
        raise ReproductionInfrastructureError("reproduction_service_failed") from None
