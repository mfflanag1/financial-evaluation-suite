"""A distinct native training adapter; frozen financial scoring is unchanged."""

__version__ = "0.1.0"
__all__ = ["FinancialPortfolioTrainingEnv", "FinancialPortfolioTrainingTaskset",
           "FinancialPortfolioTrainingHarness"]


def __getattr__(name):
    if name == "FinancialPortfolioTrainingEnv":
        from .environment import FinancialPortfolioTrainingEnv
        return FinancialPortfolioTrainingEnv
    if name == "FinancialPortfolioTrainingTaskset":
        from .task import FinancialPortfolioTrainingTaskset
        return FinancialPortfolioTrainingTaskset
    if name == "FinancialPortfolioTrainingHarness":
        from .harness import FinancialPortfolioTrainingHarness
        return FinancialPortfolioTrainingHarness
    raise AttributeError(name)
