"""Generated research pilot; independent of the frozen evaluation banks."""

__all__ = ["FinancialResearchTaskset"]


def __getattr__(name):
    if name == "FinancialResearchTaskset":
        from .prime_adapter import FinancialResearchTaskset
        return FinancialResearchTaskset
    raise AttributeError(name)
