"""Experimental portfolio tasks using the unchanged v0.2 runtime boundaries.

Only episode registration, trusted verifier staging, and reference selection
differ. Artifact parsing, public reproduction, scoring and failure handling are
inherited from the exactly pinned ``financial-research==0.2.0`` dependency.
"""

from importlib.resources import files
import json
import math
from typing import Literal
from uuid import uuid4

import verifiers.v1 as vf
from financial_research import prime_adapter as base
from financial_research.verifier import reference_submission, score_submission


class FinancialPortfolioTask(base.FinancialResearchTask):
    async def stage_verifier(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        from .generator import resolve_episode

        base._require_container(runtime)
        episode = resolve_episode(self.data.id)
        if (episode["id"] != self.data.id or episode["prompt"] != self.data.prompt
                or episode["public_files"] != self.data.public_files):
            raise ValueError("Task record differs from its registered episode")
        directory = f"/tmp/financial-verifier-{uuid4().hex}"
        trusted = files("financial_research")
        try:
            await runtime.write(f"{directory}/financial_research/__init__.py", b"")
            # The generic scorer consumes the materialized episode; no generator
            # or portfolio implementation is needed in the private runtime.
            for name in ("verifier.py", "analytics.py"):
                await runtime.write(f"{directory}/financial_research/{name}",
                                    trusted.joinpath(name).read_bytes())
            await runtime.write(f"{directory}/episode.json",
                                json.dumps(episode, allow_nan=False).encode())
            await runtime.write(f"{directory}/score.py", base._SCORE_PROGRAM.encode())
        except Exception:
            base._outcome(trace, "infrastructure_error", "verifier_staging_failed", "staging", False)
            raise RuntimeError("verifier_staging_failed") from None
        trace.state.verifier_directory = directory
        trace.state.verifier_runtime = runtime.name
        trace.state.grade = None
        trace.state.reproduction = None
        trace.state.analysis_artifact_valid = False

    async def validate(self, trace: vf.Trace, runtime: vf.Runtime) -> bool:
        from .generator import resolve_episode

        episode = resolve_episode(self.data.id)
        reference = reference_submission(episode)
        expected = score_submission(episode, reference)["reward"]
        await runtime.write("/workspace/policy.json", json.dumps(reference, allow_nan=False).encode())
        await runtime.write("/workspace/research.py", files("financial_portfolio").joinpath("reference.py").read_bytes())
        await self.finalize(trace, runtime)
        trace.state.artifacts = await vf.collect(
            runtime, self.data.artifacts, max_bytes=self.data.artifact_max_bytes)
        await vf.grade_submission(self, trace, runtime)
        return expected > 0 and math.isclose(trace.reward, expected, abs_tol=1e-12)


class PortfolioTasksetConfig(base.ResearchTasksetConfig):
    family: Literal["impact_sizing", "liquidity_tight", "risk_concentration", "no_signal"] | None = None


class FinancialPortfolioTaskset(vf.Taskset[FinancialPortfolioTask, PortfolioTasksetConfig]):
    def load(self):
        from .generator import build_episode

        for seed in range(self.config.seed_start, self.config.seed_start + self.config.num_tasks):
            episode = build_episode(seed, split=self.config.split, family=self.config.family)
            data = base.ResearchData(
                id=episode["id"], prompt=episode["prompt"], public_files=episode["public_files"],
                image=self.config.task.image, workdir="/workspace",
                network=vf.NetworkPolicyConfig(allow=[]),
                scoring=vf.ScoringConfig(mode="isolated", network=vf.NetworkPolicyConfig(allow=[])),
                artifacts=[vf.Artifact(source="/workspace/policy.json", required=False),
                           vf.Artifact(source="/workspace/research.py", required=False)],
                artifact_max_bytes=4 * 1024 * 1024,
                timeout=vf.TaskTimeout(setup=120, agent=600, finalize=30,
                                       scoring=self.config.task.scoring_seconds),
                resources=vf.TaskResources(cpu=1, memory=1),
            )
            yield FinancialPortfolioTask(data, self.config.task)
