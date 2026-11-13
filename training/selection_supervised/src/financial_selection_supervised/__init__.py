"""Whole-run supervision, separate from published financial rewards and admission."""
__version__ = "0.1.0"
__all__ = ["FinancialSelectionSupervisedEnv"]


def __getattr__(name):
    if name == "FinancialSelectionSupervisedEnv":
        from .environment import FinancialSelectionSupervisedEnv
        return FinancialSelectionSupervisedEnv
    raise AttributeError(name)
