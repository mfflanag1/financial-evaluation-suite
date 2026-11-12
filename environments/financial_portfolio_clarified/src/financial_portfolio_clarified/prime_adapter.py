"""A fixed public split convention; all financial and workflow checks are inherited."""

from __future__ import annotations

import hashlib

import verifiers.v1 as vf
from financial_portfolio.prime_adapter import FinancialPortfolioTask, PortfolioTasksetConfig
from financial_portfolio_staged import prime_adapter as staged


PROTOCOL_ID = "portfolio-clarified-v1"
CONDITION_KEY = "portfolio_condition"
CLARIFICATION = (
    "shares_per_original is the cumulative number of current shares corresponding to one "
    "original share; close is price per current share. Define adjusted_close = close * "
    "shares_per_original."
)


def clarified_prompt(public_files: dict[str, str]) -> str:
    return staged.workflow_prompt(public_files) + "\n\n" + CLARIFICATION


def _bind_condition(trace: vf.Trace) -> None:
    trace.info[CONDITION_KEY] = {
        "protocol_id": PROTOCOL_ID,
        "baseline_protocol_id": staged.PROTOCOL_ID,
        "environment": "financial-portfolio-clarified",
        "environment_version": "0.5.0",
        "clarification_sha256": hashlib.sha256(CLARIFICATION.encode()).hexdigest(),
    }


class FinancialPortfolioClarifiedTask(staged.FinancialPortfolioStagedTask):
    def _check_prompt(self) -> None:
        if self.data.system_prompt != clarified_prompt(self.data.public_files):
            raise ValueError("Clarified system instructions differ from the registered protocol")

    async def setup(self, runtime: vf.Runtime, trace: vf.Trace | None = None) -> None:
        self._check_prompt()
        await super().setup(runtime, trace)
        if trace is not None:
            _bind_condition(trace)

    async def stage_verifier(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        staged._SOLVERS.pop(trace.id, None)
        self._check_prompt()
        _bind_condition(trace)
        # Replace only the staged prompt binding; use its identical frozen parent
        # staging path without mutating shared task data or executing solver code.
        await FinancialPortfolioTask.stage_verifier(self, trace, runtime)


class FinancialPortfolioClarifiedTaskset(
        vf.Taskset[FinancialPortfolioClarifiedTask, PortfolioTasksetConfig]):
    def load(self):
        for task in staged.FinancialPortfolioStagedTaskset(self.config).load():
            data = task.data.model_copy(update={
                "system_prompt": clarified_prompt(task.data.public_files),
            })
            yield FinancialPortfolioClarifiedTask(data, task.config)
