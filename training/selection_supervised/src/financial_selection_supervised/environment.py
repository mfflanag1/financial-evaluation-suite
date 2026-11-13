"""Explicit worker readiness after the unchanged training environment starts."""
import os
from pathlib import Path

from financial_selection_training.environment import (
    FinancialSelectionTrainingEnv, FinancialSelectionTrainingEnvConfig,
)
from verifiers.v1.env import Env
from .ledger import Ledger


class FinancialSelectionSupervisedEnvConfig(FinancialSelectionTrainingEnvConfig):
    id: str = "financial-selection-supervised"


class FinancialSelectionSupervisedEnv(FinancialSelectionTrainingEnv, Env[FinancialSelectionSupervisedEnvConfig]):
    async def start(self):
        self._optimizer_observer = None
        if os.environ.get("FINANCIAL_SELECTION_OPTIMIZER_PLAN_PATH"):
            from .gpu_observer import install_worker_observer
            self._optimizer_observer = install_worker_observer(
                plan_path=Path(os.environ["FINANCIAL_SELECTION_OPTIMIZER_PLAN_PATH"]),
                evidence_dir=Path(os.environ["FINANCIAL_SELECTION_OPTIMIZER_EVIDENCE_DIR"]))
        try:
            await super().start()
            Ledger.from_env().worker_ready(os.getpid())
        except BaseException:
            if self._optimizer_observer is not None:
                self._optimizer_observer.close()
            raise

    async def stop(self):
        try:
            await super().stop()
        finally:
            observer = getattr(self, "_optimizer_observer", None)
            if observer is not None:
                observer.close()
                self._optimizer_observer = None
