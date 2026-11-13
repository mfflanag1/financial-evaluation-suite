"""Native portfolio runtime; frozen financial calculations remain unchanged."""

__version__ = "0.1.0"
__all__ = ["FinancialPortfolioRuntimeTaskset", "FinancialPortfolioRuntimeHarness"]


def __getattr__(name):
    if name == "FinancialPortfolioRuntimeHarness":
        from .harness import FinancialPortfolioRuntimeHarness
        return FinancialPortfolioRuntimeHarness
    if name == "FinancialPortfolioRuntimeTaskset":
        from .prime_adapter import FinancialPortfolioRuntimeTaskset
        return FinancialPortfolioRuntimeTaskset
    raise AttributeError(name)
