"""Prelaunch schedule and expected wire identity for one serial native run."""

import hashlib

from financial_portfolio_operations.telemetry import active_journal as provider_journal

from .telemetry import active_journal, prompt_binding


def task_binding(data, harness):
    system = "\n\n".join(text for text in (data.system_prompt, "External network access is disabled.") if text)
    messages = harness.outbound_messages(data.model_copy(update={"system_prompt": system}))
    return {"task_id": data.id,
            "public_sha256": {name: hashlib.sha256(value.encode()).hexdigest() for name, value in data.public_files.items()},
            "prompt_sha256": hashlib.sha256(data.prompt.encode()).hexdigest(),
            "system_prompt_sha256": hashlib.sha256(data.system_prompt.encode()).hexdigest(),
            "outbound_prompt": prompt_binding(messages)}


def build_run_binding(config, tasks, *, config_sha256, bootstrap_sha256=None, protocol_sha256=None):
    from verifiers.v1.configs.cli.run import RunnerConfig
    from copy import deepcopy
    from .harness import FinancialPortfolioRuntimeHarness

    resolved = RunnerConfig.model_validate(deepcopy(config))
    harness = FinancialPortfolioRuntimeHarness(resolved.env.agent.harness)
    rows = [task_binding(task.data, harness) for task in tasks]
    ids = [row["task_id"] for row in rows]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Bound schedule requires distinct tasks")
    return {"schema_version": 1, "protocol_id": "portfolio-runtime-v1",
            "bootstrap_sha256": bootstrap_sha256, "protocol_sha256": protocol_sha256,
            "capture_id": active_journal().run_id, "provider_run_id": provider_journal().run_id,
            "config_sha256": config_sha256, "expected_attempts": len(rows),
            "dry_run": config.get("dry_run", False), "model": config["model"],
            "sampling": resolved.sampling.model_dump(mode="json", exclude_none=True),
            "docker_image_id": config["env"]["agent"]["runtime"]["image"], "schedule": rows}
