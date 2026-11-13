"""A distinct native training adapter; frozen financial scoring is unchanged."""

__version__ = "0.1.0"
__all__ = ["FinancialSelectionTrainingEnv", "FinancialSelectionTrainingTaskset",
           "FinancialSelectionTrainingHarness"]


def __getattr__(name):
    if name == "FinancialSelectionTrainingEnv":
        from .environment import FinancialSelectionTrainingEnv
        return FinancialSelectionTrainingEnv
    if name == "FinancialSelectionTrainingTaskset":
        from .task import FinancialSelectionTrainingTaskset
        return FinancialSelectionTrainingTaskset
    if name == "FinancialSelectionTrainingHarness":
        from .harness import FinancialSelectionTrainingHarness
        return FinancialSelectionTrainingHarness
    raise AttributeError(name)
