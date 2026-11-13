"""Same public financial problem and scoring, with training lifecycle observations."""

import verifiers.v1 as vf
from financial_portfolio.prime_adapter import FinancialPortfolioTask, PortfolioTasksetConfig

from . import workflow
from .journal import digest, owner, record

CLARIFICATION = (
    "shares_per_original is the cumulative number of current shares corresponding to one "
    "original share; close is price per current share. Define adjusted_close = close * "
    "shares_per_original."
)


def system_prompt(public_files):
    return workflow.workflow_prompt(public_files) + "\n\n" + CLARIFICATION


class FinancialPortfolioTrainingTask(workflow.WorkflowTask):
    def check_prompt(self):
        if self.data.system_prompt != system_prompt(self.data.public_files):
            raise ValueError("Training task prompt differs from the public protocol")

    async def setup(self, runtime, trace=None):
        self.check_prompt()
        await super().setup(runtime, trace)
        if trace is not None:
            record(trace, "task_setup_complete")

    async def stage_verifier(self, trace, runtime):
        workflow._SOLVERS.pop(trace.id, None)
        self.check_prompt()
        await FinancialPortfolioTask.stage_verifier(self, trace, runtime)

    @vf.intercept
    async def checkpoint(self, request: vf.Request, trace: vf.Trace) -> vf.Request | None:
        if not request.messages or not isinstance(request.messages[-1], vf.ToolMessage):
            return await super().checkpoint(request, trace)
        result = request.messages[-1]
        position = next((i for i in range(len(request.messages) - 2, -1, -1)
                         if isinstance(request.messages[i], vf.AssistantMessage)
                         and any(c.id == result.tool_call_id for c in request.messages[i].tool_calls or [])), None)
        if position is None:
            return await super().checkpoint(request, trace)
        assistant = request.messages[position]
        call = next(c for c in assistant.tool_calls if c.id == result.tool_call_id)
        occurrence = digest([m.model_dump(mode="json") for m in request.messages[:position + 1]]) + ":" + call.id
        _, attempt = owner(trace.id)
        previous = attempt.tools.get(occurrence)
        if previous is None:
            attempt.tools[occurrence] = "gating"
            record(trace, "tool_gate_started", occurrence=occurrence, tool_name=call.name)
        elif previous == "allowed":
            attempt.tools[occurrence] = "result_observed"
            record(trace, "tool_result_observed", occurrence=occurrence, content_sha256=digest(result.content))
        replacement = await super().checkpoint(request, trace)
        if previous is None:
            status = "gate_replied" if replacement is not None else "allowed"
            attempt.tools[occurrence] = status
            record(trace, "tool_gate_terminal", occurrence=occurrence, status=status)
        return replacement

    async def _grade(self, trace, runtime):
        try:
            grade = await super()._grade(trace, runtime)
        except BaseException as error:
            record(trace, "scoring_failed", exception_type=type(error).__name__)
            raise
        _, attempt = owner(trace.id)
        completed = [event for event in attempt.events if event["event"] == "scoring_complete"]
        outcome_sha = digest(trace.info.get("financial_research_outcome"))
        if completed:
            if completed[0]["reward"] != grade["reward"] or completed[0]["outcome_sha256"] != outcome_sha:
                raise ValueError("Repeated scoring changed the financial result")
        else:
            record(trace, "scoring_complete", reward=grade["reward"], outcome_sha256=outcome_sha)
        return grade


class FinancialPortfolioTrainingTaskset(vf.Taskset[FinancialPortfolioTrainingTask, PortfolioTasksetConfig]):
    def load(self):
        for task in workflow.WorkflowTaskset(self.config).load():
            data = task.data.model_copy(update={"system_prompt": system_prompt(task.data.public_files)})
            yield FinancialPortfolioTrainingTask(data, task.config)
