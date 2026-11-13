"""Synthetic native graphs for protocol tests; these are not policy rollouts."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import verifiers.v1 as vf
from verifiers.v1.types import WeightVersionSpan

from financial_selection_training.journal import ObservedTrainClient, digest

MESSAGES = [{"role": "system", "content": "Public instructions"},
            {"role": "user", "content": "Synthetic contract fixture"}]


def response():
    return vf.Response(id="synthetic", created=0, model="fixture", finish_reason="stop",
                       message=vf.AssistantMessage(content="Synthetic output"),
                       tokens=vf.TurnTokens(prompt_ids=np.array([1, 2], dtype=np.int32),
                                           completion_ids=np.array([3, 4], dtype=np.int32),
                                           completion_logprobs=np.array([-.2, -.3], dtype=np.float32),
                                           sampler_support=vf.SamplerSupport(
                                               ids=np.array([3, 8, 4], dtype=np.int32),
                                               counts=np.array([2, 1], dtype=np.int32),
                                               replayed=np.array([True, True]),
                                               logprobs=np.array([-.2, -1.7, -.3], dtype=np.float32)),
                                           weight_versions=[WeightVersionSpan(version="fixture-v1", start=0, end=2)]))


def trace(identity="trace-one", reward=1.0):
    result = response()
    native = vf.Trace(
        id=identity, task=vf.TraceTask(type="fixture", data=vf.TaskData(id="same-task", prompt=MESSAGES[1]["content"])),
        agent=vf.AgentInfo(config=vf.AgentConfig(), trainable=True),
        nodes=[vf.MessageNode(message=vf.SystemMessage(content=MESSAGES[0]["content"]), token_ids=[1], mask=[False]),
               vf.MessageNode(parent=0, message=vf.UserMessage(content=MESSAGES[1]["content"]), token_ids=[2], mask=[False]),
               vf.MessageNode(parent=1, message=result.message, token_ids=[3, 4], mask=[True, True],
                              logprobs=[-.2, -.3], sampled=True,
                              sampler_support=result.tokens.sampler_support,
                              weight_versions=result.tokens.weight_versions)],
        calls=[vf.ModelCall(node=2, model="fixture")], is_completed=True, ok=True,
        rewards={"selection_reward": vf.Reward(score=reward)},
        info={"financial_selection_outcome": {"status": "scored" if reward else "candidate_failure",
                                             "reason": "fixture", "reproduction_verified": bool(reward)}})

    native.metrics = {"candidate_failure": float(not reward), "reproduction_verified": float(bool(reward)),
        "packets_scored": 3.0 if reward else 0.0, "exact_success_fraction": float(reward == 1),
        "all_packets_correct": float(reward == 1), "current_exact_success": float(reward == 1),
        "fresh_exact_success_fraction": float(reward == 1)}
    native.info["scoring_diagnostics"] = {"financial_selection_outcome": dict(native.info["financial_selection_outcome"])}
    return native


def lifecycle(journal, attempt):
    for kind, runtime in (("solver_setup_complete", "solver"), ("artifacts_captured", "solver"),
                          ("verifier_setup_complete", "verifier"), ("verifier_staged", "verifier")):
        journal.emit(attempt, kind, runtime_name_sha256=digest(runtime))


async def completed(journal, identity="trace-one", reward=1.0):
    native = trace(identity, reward)
    attempt = journal.register(native, episode_id="episode-" + identity, episode_attempt=1, expected_messages=MESSAGES, model="fixture")
    journal.bind(native, MESSAGES)
    journal.emit(attempt, "harness_started")
    client = ObservedTrainClient(SimpleNamespace(get_response=AsyncMock(return_value=response())))
    await client.get_response(None, {"messages": deepcopy(MESSAGES), "model": "fixture"}, None, session_id=native.id)
    journal.emit(attempt, "harness_terminal", outcome="returned", exit_code=0)
    journal.emit(attempt, "runtime_health", alive=True)
    lifecycle(journal, attempt)
    journal.emit(attempt, "scoring_complete", reward=reward, grade_sha256=digest({"reward": float(reward), "metrics": dict(native.metrics)}),
                 outcome_sha256=digest(native.info["financial_selection_outcome"]))
    return native, attempt


def episode(native):
    return vf.Episode(task=native.task, env=vf.EnvInfo(id="financial-selection-training"),
                      traces=[native], ok=True)
