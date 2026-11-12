"""Staged harness with opt-in trusted-host request accounting; solver behavior unchanged."""

from financial_portfolio_staged.harness import (
    FinancialPortfolioStagedHarness,
    FinancialPortfolioStagedHarnessConfig,
)
from verifiers.v1.harness import Harness

from .telemetry import active_journal


class FinancialPortfolioOperationsHarnessConfig(FinancialPortfolioStagedHarnessConfig):
    id: str = "financial-portfolio-operations"


class FinancialPortfolioOperationsHarness(
    FinancialPortfolioStagedHarness, Harness[FinancialPortfolioOperationsHarnessConfig]
):
    async def launch(self, ctx, trace, runtime, endpoint, secret, mcp_urls, data,
                     tool_interception_url=None):
        active_journal().register(trace_id=trace.id, task_id=trace.task.data.id, model=ctx.model)
        return await super().launch(ctx, trace, runtime, endpoint, secret, mcp_urls, data,
                                    tool_interception_url=tool_interception_url)
