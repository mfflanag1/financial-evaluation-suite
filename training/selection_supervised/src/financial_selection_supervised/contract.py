"""Narrow one-step local contract; source drift and unreviewed configuration fail shut."""
from copy import deepcopy
import hashlib
from importlib import metadata
from importlib.resources import files
import json
import os
from pathlib import Path
import re

KEY_ENV = "FINANCIAL_SELECTION_KEY"
OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
               "WANDB_MODE": "disabled", "HF_HUB_DISABLE_TELEMETRY": "1"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def template():
    return json.loads(files("financial_selection_supervised").joinpath("contract.json").read_text())


def make_config(model, image, root):
    config = template()
    config["model"]["name"] = str(Path(model))
    config["output_dir"] = str(Path(root) / "native")
    env = config["orchestrator"]["train"]["source"][0]["env"]
    env["journal_root"] = str(Path(root) / "journals")
    env["agent"]["runtime"]["image"] = image
    env["taskset"]["task"]["image"] = image
    return config


def validate_config(raw, root, *, require_resources=True):
    root = Path(root)
    if not root.is_absolute() or root.resolve() != root or root.is_symlink():
        raise ValueError("Supervisor root must be an absolute canonical path")
    if any((parent / ".git").exists() for parent in (root, *root.parents)):
        raise ValueError("Private run artifacts must be outside Git checkouts")
    try:
        model = raw["model"]["name"]
        image = raw["orchestrator"]["train"]["source"][0]["env"]["agent"]["runtime"]["image"]
    except (KeyError, TypeError, IndexError):
        raise ValueError("Incomplete supervised contract") from None
    if not isinstance(model, str) or not Path(model).is_absolute() or Path(model).resolve() != Path(model):
        raise ValueError("A canonical absolute local model path is required")
    if not isinstance(image, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", image) is None or image == "sha256:" + "0" * 64:
        raise ValueError("A concrete immutable local Docker image is required")
    # Equality is intentionally not permissive Pydantic coercion: booleans, int/float
    # substitutions, missing controls and extra native options are rejected.
    expected = make_config(model, image, root)
    if json.dumps(raw, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError("Configuration differs from the supervised v1 allowlist")
    if require_resources:
        directory = Path(model)
        if not (directory / "config.json").is_file() or not any((directory / p).is_file() for p in ("tokenizer.json", "tokenizer.model")):
            raise ValueError("Local policy and tokenizer files must already exist")
        if not any(directory.glob("*.safetensors")) and not any(directory.glob("pytorch_model*.bin")):
            raise ValueError("Local policy weights must already exist; downloading is disabled")
    key = os.environ.get(KEY_ENV)
    if key and key.lower() in json.dumps(raw).lower():
        raise ValueError("Generation key cannot appear in configuration")
    from prime_rl.configs.rl import RLConfig
    config = RLConfig.model_validate(deepcopy(raw))
    if (config.deployment.type != "single_node" or config.slurm is not None or config.max_steps != 1
            or config.resume is not None or config.dry_run or config.dashboard
            or config.orchestrator.env_worker_pool_replicas is not None):
        raise ValueError("Native resolution escaped the supervised local contract")
    return config


def sanitized_environment(source, key=None):
    """Remove the generation key and aliases from every untrusted component env."""
    key = key if key is not None else source.get(KEY_ENV)
    return {name: value for name, value in source.items()
            if name != KEY_ENV and not (key and key.lower() in value.lower())}


def verify_sources():
    from financial_selection_training.environment import verify_native_sources
    verify_native_sources()
    package = files("financial_selection_supervised")
    manifest_bytes = package.joinpath("source_bindings.json").read_bytes()
    bindings = json.loads(manifest_bytes)
    for distribution, expected in bindings["distributions"].items():
        if metadata.version(distribution) != expected:
            raise RuntimeError("Unexpected installed distribution: " + distribution)
    for path, expected in bindings["native_surfaces"].items():
        module, relative = path.split("/", 1)
        if hashlib.sha256(files(module).joinpath(relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError("Supervisor native interception source drift: " + path)
    for path, expected in bindings["supervisor_sources"].items():
        if hashlib.sha256(package.joinpath(path).read_bytes()).hexdigest() != expected:
            raise RuntimeError("Supervisor source drift: " + path)
    return hashlib.sha256(manifest_bytes).hexdigest()
