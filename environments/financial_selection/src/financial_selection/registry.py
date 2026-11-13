"""Bounded deterministic worker registry; only opaque identities cross the wire.

There is no process-local registration prerequisite. A newly spawned trusted
worker recovers an identity by recomputing at most 960 keyed digests, then
regenerates the public current packet and two disjoint fresh-index packets.
"""
from dataclasses import dataclass
import hashlib
import hmac
import json
import os
import re

from .generator import generate


KEY_ENV = "FINANCIAL_SELECTION_KEY"
SPLITS = ("train", "dev", "final")
CONDITIONS = ("mixed", "positive", "negative", "unsupported", "sparse")
CURRENT_INDEX_COUNT = 64
FRESH_INDEX_START = 128


class RegistryError(RuntimeError):
    """Private registry service/configuration error, never a candidate zero."""


def worker_key():
    encoded = os.environ.get(KEY_ENV, "")
    if (not isinstance(encoded, str) or len(encoded) % 2
            or re.fullmatch(r"[0-9a-fA-F]{64,1024}", encoded) is None):
        raise RegistryError("worker_key_unavailable")
    return bytes.fromhex(encoded)


def _selection(split, index, condition):
    if (split not in SPLITS or condition not in CONDITIONS or type(index) is not int
            or not 0 <= index < CURRENT_INDEX_COUNT):
        raise RegistryError("task_selection_outside_registry")


def _identity(key, split, index, condition):
    message = json.dumps(["selection-episode-v1", split, index, condition], separators=(",", ":")).encode()
    return hmac.new(key, message, hashlib.sha256).hexdigest()


def public_episode(split, index, condition="mixed"):
    """Return public data only; no hidden population survives this function."""
    _selection(split, index, condition)
    key = worker_key()
    episode = generate(key, split, index, condition=condition)
    if not hmac.compare_digest(episode.identity, _identity(key, split, index, condition)):
        raise RegistryError("generator_identity_contract_changed")
    return episode.identity, episode.public_files


@dataclass(frozen=True, repr=False)
class ResolvedPackets:
    public_files: dict[str, str]
    fresh_packets: list[dict[str, str]]


def resolve(identity):
    """Recover a registered HMAC without revealing its generator coordinates."""
    if not isinstance(identity, str) or re.fullmatch(r"[0-9a-f]{64}", identity) is None:
        raise RegistryError("unknown_task_identity")
    key = worker_key()
    matched = None
    for split in SPLITS:
        for index in range(CURRENT_INDEX_COUNT):
            for condition in CONDITIONS:
                if hmac.compare_digest(identity, _identity(key, split, index, condition)):
                    matched = split, index, condition
    if matched is None:
        raise RegistryError("unknown_task_identity")
    split, index, condition = matched
    current = generate(key, split, index, condition=condition)
    fresh = [generate(key, split, FRESH_INDEX_START + 2 * index + offset,
                      condition=condition).public_files for offset in range(2)]
    return ResolvedPackets(current.public_files, fresh)
