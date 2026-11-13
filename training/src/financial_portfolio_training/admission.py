"""Native drop admission, preserving every financial reward and diagnostic."""

import math
import numpy as np
import verifiers.v1 as vf

from .journal import INFO_KEY, _array_binding, completion_summary, digest, prompt_binding


def valid_graph(trace):
    """Native graph nodes are appended after their parents, never cyclic."""
    return all(node.parent is None or (type(node.parent) is int and 0 <= node.parent < index)
               for index, node in enumerate(trace.nodes))


def _controlled_stop(trace):
    workflow = trace.info.get("portfolio_workflow", {})
    outcome = trace.info.get("financial_research_outcome", {})
    stages = ["ingestion", "trials", "submission"]
    if trace.stop_condition == "workflow_complete":
        checkpoints = [row for row in workflow.get("checkpoints", []) if row.get("valid") is True]
        return (workflow.get("completed") == stages and len(checkpoints) == 3
                and [row.get("stage") for row in checkpoints] == stages
                and all(row.get("valid") is True for row in checkpoints)
                and bool(workflow.get("submission_sha256")))
    if outcome.get("status") != "candidate_failure":
        return False
    if trace.stop_condition == "candidate_protocol_failure":
        return workflow.get("protocol_failure", {}).get("reason") == "malformed_tool_arguments"
    if trace.stop_condition == "candidate_workflow_failure":
        node = trace.calls[-1].node if trace.calls else None
        return (workflow.get("completed") != stages and bool(trace.calls)
                and type(node) is int and 0 <= node < len(trace.nodes)
                and isinstance(trace.nodes[node].message, vf.AssistantMessage)
                and not trace.nodes[node].message.tool_calls)
    return False


def decision(trace, attempt):
    """Only complete, bound, resolved financial executions may enter training."""
    if not trace.ok or not trace.is_completed or trace.is_timeout or trace.is_dropped:
        return False, "native_trace_unresolved"
    if not valid_graph(trace):
        return False, "native_graph_unresolved"
    info = trace.info.get(INFO_KEY, {})
    if (info.get("task_sha256") != attempt.task_sha256
            or digest(trace.task.data.model_dump(mode="json")) != attempt.task_sha256
            or trace.task.data.id != attempt.task_id or info.get("prompt_binding") != attempt.prompt
            or attempt.prompt is None):
        return False, "task_or_prompt_binding_mismatch"
    initial = [{"role": node.message.role, "content": node.message.content} for node in trace.nodes[:2]]
    try:
        if prompt_binding(initial) != attempt.prompt:
            return False, "native_prompt_binding_mismatch"
    except (KeyError, ValueError, TypeError, StopIteration):
        return False, "native_prompt_binding_mismatch"
    events = attempt.events
    terminal = [event for event in events if event["event"] == "harness_terminal"]
    health = [event for event in events if event["event"] == "runtime_health"]
    scores = [event for event in events if event["event"] == "scoring_complete"]
    if len(terminal) != 1 or terminal[0].get("outcome") != "returned":
        return False, "harness_unresolved"
    if terminal[0].get("exit_code") != 0 and not _controlled_stop(trace):
        return False, "harness_exit_unclassified"
    if len(health) != 1 or health[0].get("alive") is not True:
        return False, "runtime_health_unresolved"
    if any(state not in ("gate_replied", "result_observed") for state in attempt.tools.values()):
        return False, "tool_execution_unresolved"
    checks = [event for event in events if event["event"] == "prompt_checked"]
    if (not attempt.requests or len(checks) != len(attempt.requests)
            or len(trace.calls) != len(attempt.requests)
            or any(event.get("valid") is not True for event in checks)
            or any(request["outcome"] != "success" for request in attempt.requests)):
        return False, "provider_requests_unreconciled"
    outcome = trace.info.get("financial_research_outcome", {})
    if outcome.get("status") not in ("passed", "candidate_failure"):
        return False, "financial_outcome_unresolved"
    if trace.info.get("portfolio_workflow", {}).get("model_free_reference"):
        return False, "validation_only"
    if (len(scores) != 1 or any(event["event"] == "scoring_failed" for event in events)
            or scores[0].get("outcome_sha256") != digest(outcome)
            or not trace.rewards or any(reward is None for reward in trace.rewards.values())
            or not math.isfinite(trace.reward) or scores[0].get("reward") != trace.reward):
        return False, "scoring_unreconciled"
    sampled = 0
    for node in trace.nodes:
        ids, mask, logprobs = map(np.asarray, (node.token_ids, node.mask, node.logprobs))
        count = int(np.count_nonzero(mask))
        if (ids.ndim != 1 or mask.ndim != 1 or logprobs.ndim != 1 or len(ids) != len(mask)
                or len(logprobs) != count or not np.all(np.isfinite(logprobs))):
            return False, "training_arrays_unaligned"
        support = node.sampler_support
        if support is not None and (len(support.counts) != count or len(support.replayed) != count
                                    or int(np.asarray(support.counts).sum()) != len(support.ids)):
            return False, "sampler_support_unaligned"
        sampled += count
    joined = set()
    for call, request in zip(trace.calls, attempt.requests):
        if (type(call.node) is not int or not 0 <= call.node < len(trace.nodes)
                or call.node in joined or call.model != attempt.model or call.error is not None):
            return False, "native_call_unreconciled"
        node = trace.nodes[call.node]
        joined.add(call.node)
        try:
            # Native graph.commit_turn stores the exact prompt in the ordered
            # ancestor path and this call node's unsampled generation prefix.
            ancestors, parent, seen = [], node.parent, {call.node}
            while parent is not None:
                if type(parent) is not int or not 0 <= parent < call.node or parent in seen:
                    return False, "native_prompt_tokens_unreconciled"
                seen.add(parent)
                ancestors.append(trace.nodes[parent])
                parent = trace.nodes[parent].parent
            mask = np.asarray(node.mask)
            if np.any(np.diff(mask.astype(np.int8)) < 0):
                return False, "native_prompt_tokens_unreconciled"
            prompt = np.concatenate([np.asarray(ancestor.token_ids) for ancestor in reversed(ancestors)]
                                    + [np.asarray(node.token_ids)[~mask]])
            if request.get("prompt_ids") != _array_binding(prompt):
                return False, "native_prompt_tokens_unreconciled"
            observed = completion_summary(np.asarray(node.token_ids)[np.asarray(node.mask)],
                                          node.logprobs, node.sampler_support, node.weight_versions)
        except (TypeError, ValueError, IndexError):
            return False, "training_response_unreconciled"
        if any(request.get(key) != value for key, value in observed.items()):
            return False, "training_response_unreconciled"
    if any(np.any(node.mask) and index not in joined for index, node in enumerate(trace.nodes)):
        return False, "sampled_node_unreconciled"
    if not trace.agent.trainable or not sampled:
        return False, "no_trainable_tokens"
    return True, outcome["status"]


def apply(trace, journal):
    attempt = journal.attempts.get(trace.id)
    if attempt is None or attempt.terminal:
        raise RuntimeError("Training admission has no unique live registered attempt")
    original = trace.rewards.copy()
    admitted, reason = decision(trace, attempt)
    trace.info[INFO_KEY]["admission"] = {"admitted": admitted, "reason": reason}
    if not admitted:
        trace.drop("portfolio-training-v1: " + reason)
    journal.finish(trace, admitted=admitted, reason=reason)
    if trace.rewards != original:
        raise RuntimeError("Training admission changed native financial rewards")
    return admitted
