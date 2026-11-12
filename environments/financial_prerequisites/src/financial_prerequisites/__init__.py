"""Small deterministic prerequisites, separate from the financial scorebanks."""

__all__ = ["FinancialPrerequisitesTaskset"]


def __getattr__(name):
    if name == "FinancialPrerequisitesTaskset":
        from .prime_adapter import FinancialPrerequisitesTaskset
        return FinancialPrerequisitesTaskset
    raise AttributeError(name)
