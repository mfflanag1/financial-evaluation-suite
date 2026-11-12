"""One prompt clarification around the frozen staged portfolio environment."""

__all__ = ["FinancialPortfolioClarifiedTaskset"]


def __getattr__(name):
    if name == "FinancialPortfolioClarifiedTaskset":
        from .prime_adapter import FinancialPortfolioClarifiedTaskset
        return FinancialPortfolioClarifiedTaskset
    raise AttributeError(name)
