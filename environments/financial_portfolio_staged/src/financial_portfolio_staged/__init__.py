"""Separate staged workflow; frozen financial tasks and scoring are dependencies."""

__all__ = ["FinancialPortfolioStagedTaskset", "FinancialPortfolioStagedHarness"]


def __getattr__(name):
    if name == "FinancialPortfolioStagedTaskset":
        from .prime_adapter import FinancialPortfolioStagedTaskset
        return FinancialPortfolioStagedTaskset
    if name == "FinancialPortfolioStagedHarness":
        from .harness import FinancialPortfolioStagedHarness
        return FinancialPortfolioStagedHarness
    raise AttributeError(name)
