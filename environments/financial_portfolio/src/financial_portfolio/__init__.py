"""Experimental portfolio environment, separate from the frozen research pilot."""

__all__ = ["FinancialPortfolioTaskset"]


def __getattr__(name):
    if name == "FinancialPortfolioTaskset":
        from .prime_adapter import FinancialPortfolioTaskset
        return FinancialPortfolioTaskset
    raise AttributeError(name)
