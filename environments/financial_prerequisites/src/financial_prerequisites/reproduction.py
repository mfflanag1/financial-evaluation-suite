"""Strict output comparison and fresh public-only candidate execution."""

import asyncio
import base64
import json
import math
import re

from financial_research.reproduction import (
    MAX_POLICY_BYTES, MAX_SOURCE_BYTES, ReproductionInfrastructureError,
    _CONTROLLER, fresh_config,
)


def parse_output(payload):
    """Reject duplicate keys, nonfinite numbers, and non-object roots."""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate_key")
            result[key] = value
        return result

    def reject(value):
        raise ValueError("nonfinite_number")

    if type(payload) is not bytes or len(payload) > MAX_POLICY_BYTES:
        raise ValueError("invalid_output_size")
    value = json.loads(payload, object_pairs_hook=unique, parse_constant=reject)
    if type(value) is not dict:
        raise ValueError("output_not_object")
    pending = [value]
    while pending:
        item = pending.pop()
        if type(item) is dict:
            pending.extend(item.values())
        elif type(item) is list:
            pending.extend(item)
        elif type(item) is float and not math.isfinite(item):
            raise ValueError("nonfinite_number")
    return value


def outputs_match(actual, expected):
    """Compare exact shape and discrete types; numeric quantities use 1e-9."""
    if type(expected) is dict:
        return (type(actual) is dict and actual.keys() == expected.keys()
                and all(outputs_match(actual[key], value) for key, value in expected.items()))
    if type(expected) is list:
        return (type(actual) is list and len(actual) == len(expected)
                and all(outputs_match(a, b) for a, b in zip(actual, expected)))
    if type(expected) is float:
        try:
            return (type(actual) in (int, float) and math.isfinite(actual)
                    and math.isfinite(expected) and abs(actual - expected) <= 1e-9)
        except OverflowError:
            return False
    return type(actual) is type(expected) and actual == expected


def valid_public_files(public_files):
    return (type(public_files) is dict and bool(public_files)
            and all(type(name) is str and re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*", name)
                    and name not in {"research.py", "policy.json"} and type(content) is str
                    for name, content in public_files.items()))


async def reproduce(public_files, analysis, expected, *, runtime_config, timeout_seconds=10):
    """Execute only in a new container; return a reason, or raise sanitized infra."""
    from verifiers.v1.runtimes import provision_runtime

    if not valid_public_files(public_files):
        raise ReproductionInfrastructureError("invalid_public_inputs")
    if (type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds)
            or not 0 < timeout_seconds <= 30):
        raise ReproductionInfrastructureError("invalid_reproduction_timeout")
    if type(analysis) is not bytes or not 0 < len(analysis) <= MAX_SOURCE_BYTES:
        return "invalid_analysis_artifact"
    try:
        if not analysis.decode("utf-8").strip():
            return "invalid_analysis_artifact"
    except UnicodeError:
        return "invalid_analysis_artifact"
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
            await runtime.write("/workspace/research.py", analysis)
            await runtime.backend.prepare_execution([])
            async with asyncio.timeout(timeout_seconds + 15):
                completed = await runtime.run([
                    "python", "-I", "/opt/financial-reproduction/controller.py", str(timeout_seconds)], {})
            if completed.exit_code:
                raise ReproductionInfrastructureError("reproduction_controller_failed")
            captured = json.loads(await runtime.read(
                "/opt/financial-reproduction/result.json", max_bytes=3 * 1024 * 1024))
            if type(captured) is not dict:
                raise ReproductionInfrastructureError("invalid_controller_result")
            if captured.get("reason") != "output_created":
                if captured.get("reason") not in {
                    "analysis_exit", "analysis_timeout", "output_not_regular", "output_too_large",
                    "output_missing_or_unsafe",
                }:
                    raise ReproductionInfrastructureError("invalid_controller_result")
                return captured["reason"]
            try:
                actual = parse_output(base64.b64decode(captured["policy"], validate=True))
                matched = outputs_match(actual, expected)
            except (ValueError, TypeError, KeyError, UnicodeError, OverflowError, RecursionError):
                return "invalid_reproduced_policy"
            return "passed" if matched else "output_mismatch"
    except ReproductionInfrastructureError:
        raise
    except Exception:
        raise ReproductionInfrastructureError("reproduction_service_failed") from None
