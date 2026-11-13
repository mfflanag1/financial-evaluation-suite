"""Per-worker setup and fail-closed native trace admission before worker delivery."""

import asyncio
import hashlib
from importlib.resources import files
import json
from pathlib import Path
import re
from uuid import uuid4

from pydantic import field_validator
from verifiers.v1.configs.env import SingleAgentEnvConfig
from verifiers.v1.configs.client import TrainClientConfig
from verifiers.v1.env import Env

from . import admission
from .journal import WorkerJournal, install, uninstall


def verify_native_sources():
    pins = json.loads(files("financial_portfolio_training").joinpath("source_bindings.json").read_text())
    for relative, expected in pins["native_sources"].items():
        package, remainder = relative.split("/", 1)
        if hashlib.sha256(files(package).joinpath(remainder).read_bytes()).hexdigest() != expected:
            raise RuntimeError("Native trainer source differs from pinned contract: " + relative)
    for relative, expected in pins["financial_sources"].items():
        if relative.startswith(("environments/financial_research/", "environments/financial_portfolio/")):
            package, remainder = relative.split("/src/", 1)[1].split("/", 1)
            if hashlib.sha256(files(package).joinpath(remainder).read_bytes()).hexdigest() != expected:
                raise RuntimeError("Financial dependency differs from pinned contract: " + relative)


class FinancialPortfolioTrainingEnvConfig(SingleAgentEnvConfig):
    id: str = "financial-portfolio-training"
    journal_root: Path

    @field_validator("journal_root")
    @classmethod
    def private_absolute_root(cls, value):
        if not value.is_absolute():
            raise ValueError("journal_root must be an absolute host-private path")
        return value


class FinancialPortfolioTrainingEnv(Env[FinancialPortfolioTrainingEnvConfig]):
    async def start(self):
        if getattr(self, "journal", None) is not None:
            raise RuntimeError("Training worker is already started")
        verify_native_sources()
        agent = self.config.agent
        if (self.config.retries.max_retries or agent.retries.max_retries or agent.scoring.retries.max_retries
                or agent.scoring.mode != "isolated" or agent.scoring.runtime is not None
                or agent.scoring.env or agent.scoring.network is not None
                or agent.model is not None or agent.client is not None or agent.sampling is not None
                or agent.runtime.type != "docker" or agent.runtime.allow or agent.runtime.mounts
                or not agent.network_notice):
            raise ValueError("Training v1 requires isolated Docker, no inference overrides and zero retries")
        image = agent.runtime.image
        if (not isinstance(image, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", image) is None
                or image == "sha256:" + "0" * 64 or image != self.config.taskset.task.image):
            raise ValueError("Training solver and verifier require the same concrete immutable image ID")
        self.journal = WorkerJournal(self.config.journal_root)
        self._active_episodes = set()
        self._stopping = False
        install(self.journal)

    async def stop(self):
        journal = getattr(self, "journal", None)
        if journal is not None:
            self._stopping = True
            active = [task for task in self._active_episodes if task is not asyncio.current_task() and not task.done()]
            for task in active:
                if not task.cancelling():
                    task.cancel()
            if active:
                done, pending = await asyncio.wait(active, timeout=30)
                await asyncio.gather(*done, return_exceptions=True)
                if pending:
                    # Keep the journal available to unfinished finalizers. A
                    # worker shutdown is failed until they can be accounted.
                    raise RuntimeError("Training worker episode finalizers did not finish during shutdown")
            uninstall(journal)
            self.journal = None

    async def run(self, task, agents):
        await agents.agent.run(task)

    async def finalize(self, task, episode):
        if getattr(self, "journal", None) is None:
            raise RuntimeError("Training worker journal is absent")
        if len(episode.traces) != 1:
            raise RuntimeError("Training v1 expects exactly one trace per episode")
        for trace in episode.traces:
            admission.apply(trace, self.journal)

    async def run_episode(self, task, ctx, *, on_trace=None, on_discard=None,
                          episode_id=None, attempt=1):
        if getattr(self, "journal", None) is None or self._stopping:
            raise RuntimeError("Training worker must start before episodes")
        if (not isinstance(ctx.client, TrainClientConfig) or ctx.sampling.top_p != 1.0
                or getattr(ctx.sampling, "top_k", None) not in (None, -1)
                or getattr(ctx.sampling, "min_p", 0.0) not in (None, 0.0)):
            raise ValueError("Training v1 requires native train client and untruncated sampling")
        episode_id = episode_id or uuid4().hex
        journal = self.journal
        current = asyncio.current_task()
        self._active_episodes.add(current)
        minted = []

        def register(trace):
            data = trace.task.data
            system = "\n\n".join(p for p in (data.system_prompt, "External network access is disabled.") if p)
            messages = self._harnesses["agent"].outbound_messages(data.model_copy(update={"system_prompt": system}))
            journal.register(trace, episode_id=episode_id, episode_attempt=attempt,
                             expected_messages=messages, model=ctx.model)
            minted.append(trace)
            if on_trace is not None:
                on_trace(trace)

        try:
            episode = await super().run_episode(task, ctx, on_trace=register, on_discard=on_discard,
                                                episode_id=episode_id, attempt=attempt)
            if any(not admission.valid_graph(trace) for trace in episode.traces):
                # Env.finalize exceptions become episode diagnostics upstream.
                # A malformed graph must instead produce a worker error reply:
                # serializing its computed branches can otherwise loop forever.
                raise ValueError("Training episode has an invalid native token graph")
            return episode
        finally:
            # Episode failure/cancellation can bypass Env.finalize. Never leave
            # such an attempt without a terminal accounting decision.
            try:
                for trace in minted:
                    if trace.id in journal.attempts:
                        trace.drop("portfolio-training-v1: episode_unresolved")
                        journal.finish(trace, admitted=False, reason="episode_unresolved")
            finally:
                self._active_episodes.discard(current)
