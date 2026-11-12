"""Opt-in host instrumentation around the frozen portfolio environment."""

__all__ = ["FinancialPortfolioOperationsHarness"]


def __getattr__(name):
    if name == "FinancialPortfolioOperationsHarness":
        from .harness import FinancialPortfolioOperationsHarness
        return FinancialPortfolioOperationsHarness
    raise AttributeError(name)
