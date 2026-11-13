"""Host tool lifecycle observations around the unchanged clarified financial task."""

import verifiers.v1 as vf
from financial_portfolio.prime_adapter import PortfolioTasksetConfig
from financial_portfolio_clarified.prime_adapter import (
    FinancialPortfolioClarifiedTask, FinancialPortfolioClarifiedTaskset,
)

from .telemetry import INFO_KEY, digest, record


class FinancialPortfolioRuntimeTask(FinancialPortfolioClarifiedTask):
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
        state = trace.info.setdefault(INFO_KEY, {}).setdefault("tools", {})
        previous = state.get(occurrence)
        if previous is None:
            state[occurrence] = {"status": "gating", "tool_call_id": call.id, "tool_name": call.name}
            record(trace, "tool_gate_started", occurrence=occurrence, tool_name=call.name)
        elif previous["status"] == "allowed":
            # The native session gates each call exactly once. A subsequent hook
            # invocation is the harness result, including a legitimately empty one.
            previous["status"] = "result_observed"
            record(trace, "tool_result_observed", occurrence=occurrence,
                   content_sha256=digest(result.content))
        replacement = await super().checkpoint(request, trace)
        if previous is None:
            state[occurrence]["status"] = "gate_replied" if replacement is not None else "allowed"
            record(trace, "tool_gate_terminal", occurrence=occurrence, status=state[occurrence]["status"])
        return replacement


class FinancialPortfolioRuntimeTaskset(vf.Taskset[FinancialPortfolioRuntimeTask, PortfolioTasksetConfig]):
    def load(self):
        for task in FinancialPortfolioClarifiedTaskset(self.config).load():
            yield FinancialPortfolioRuntimeTask(task.data, task.config)
