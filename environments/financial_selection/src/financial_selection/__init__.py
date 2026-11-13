"""Synthetic selection and funded allocation, with an optional native taskset."""

__version__ = "0.1.0"
__all__ = ['FinancialSelectionTaskset']


def __getattr__(name):
    if name == 'FinancialSelectionTaskset':
        from .prime_adapter import FinancialSelectionTaskset
        return FinancialSelectionTaskset
    raise AttributeError(name)
