"""Unchanged selection finance, with worker-owned lifecycle observations."""

import verifiers.v1 as vf
from financial_selection.prime_adapter import (
    FinancialSelectionTask, FinancialSelectionTaskset, SelectionTasksetConfig,
)

from .journal import INFO_KEY, digest, owner, record


class FinancialSelectionTrainingTask(FinancialSelectionTask):
    async def setup(self, runtime, trace=None):
        await super().setup(runtime, trace)
        if trace is not None:
            stage = "verifier" if trace.state.solver_runtime is not None else "solver"
            record(trace, stage + "_setup_complete", runtime_name_sha256=digest(runtime.name))

    async def finalize(self, trace, runtime):
        await super().finalize(trace, runtime)
        record(trace, "artifacts_captured", runtime_name_sha256=digest(runtime.name))

    async def stage_verifier(self, trace, runtime):
        await super().stage_verifier(trace, runtime)
        record(trace, "verifier_staged", runtime_name_sha256=digest(runtime.name))

    @vf.intercept
    async def observe_tool(self, request: vf.Request, trace: vf.Trace) -> vf.Request | None:
        """Observe native pre-execution gates and later results without replacing them."""
        if not request.messages or not isinstance(request.messages[-1], vf.ToolMessage):
            return None
        result = request.messages[-1]
        position = next((i for i in range(len(request.messages) - 2, -1, -1)
                         if isinstance(request.messages[i], vf.AssistantMessage)
                         and any(c.id == result.tool_call_id for c in request.messages[i].tool_calls or [])), None)
        if position is None:
            return None
        assistant = request.messages[position]
        call = next(c for c in assistant.tool_calls if c.id == result.tool_call_id)
        occurrence = digest([m.model_dump(mode="json") for m in request.messages[:position + 1]]) + ":" + call.id
        _, attempt = owner(trace.id)
        previous = attempt.tools.get(occurrence)
        if previous is None:
            attempt.tools[occurrence] = "allowed"
            record(trace, "tool_gate_started", occurrence=occurrence, tool_name=call.name)
            record(trace, "tool_gate_terminal", occurrence=occurrence, status="allowed")
        elif previous == "allowed":
            attempt.tools[occurrence] = "result_observed"
            record(trace, "tool_result_observed", occurrence=occurrence, content_sha256=digest(result.content))
        return None

    async def _grade(self, trace, runtime):
        try:
            grade = await super()._grade(trace, runtime)
        except BaseException as error:
            record(trace, "scoring_failed", exception_type=type(error).__name__)
            raise
        _, attempt = owner(trace.id)
        completed = [event for event in attempt.events if event["event"] == "scoring_complete"]
        observed = {"reward": grade["reward"], "grade_sha256": digest(grade),
                    "outcome_sha256": digest(trace.info.get("financial_selection_outcome"))}
        if completed:
            if any(completed[0].get(key) != value for key, value in observed.items()):
                raise ValueError("Repeated scoring changed the financial result")
        else:
            record(trace, "scoring_complete", **observed)
        return grade

    async def validate(self, trace, runtime):
        trace.info.setdefault(INFO_KEY, {})["model_free_reference"] = True
        return await super().validate(trace, runtime)


class FinancialSelectionTrainingTaskset(vf.Taskset[FinancialSelectionTrainingTask, SelectionTasksetConfig]):
    def load(self):
        for task in FinancialSelectionTaskset(self.config).load():
            yield FinancialSelectionTrainingTask(task.data, task.config)
