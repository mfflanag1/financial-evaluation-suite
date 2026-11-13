"""Idempotent optional guidance and exact model-boundary prompt verification."""

import asyncio
import hashlib

from pydantic import field_validator
from verifiers.v1.harness import Harness
from verifiers.v1.harnesses.bash.harness import BASH_SYSTEM_PROMPT, EDIT_SYSTEM_PROMPT
from verifiers.v1.harnesses.utils.launch import launch_chat_program
from financial_portfolio_staged.harness import (
    FinancialPortfolioStagedHarness, FinancialPortfolioStagedHarnessConfig,
    chat_program_source,
)
from financial_portfolio_operations.telemetry import active_journal as request_journal

from .telemetry import active_journal, record


GUIDANCE_MARKER = "\n\n<portfolio-runtime-guidance>\n"


def append_guidance(system, guidance):
    """The coordinator may resolve then copy this system prompt before launch."""
    if not guidance:
        return system
    addition = GUIDANCE_MARKER + guidance + "\n</portfolio-runtime-guidance>"
    if GUIDANCE_MARKER in (system or ""):
        if (system or "").count(GUIDANCE_MARKER) != 1 or not system.endswith(addition):
            raise ValueError("Conflicting runtime guidance in system prompt")
        return system
    return (system or "") + addition


class FinancialPortfolioRuntimeHarnessConfig(FinancialPortfolioStagedHarnessConfig):
    id: str = "financial-portfolio-runtime"
    guidance: str = ""

    @field_validator("guidance")
    @classmethod
    def bounded_guidance(cls, value):
        if len(value.encode()) > 8192 or "portfolio-runtime-guidance>" in value:
            raise ValueError("Guidance must be bounded plain text without reserved delimiters")
        return value


class FinancialPortfolioRuntimeHarness(
        FinancialPortfolioStagedHarness, Harness[FinancialPortfolioRuntimeHarnessConfig]):
    SUPPORTS_RESUME = False

    def resolve_prompt(self, data):
        system, prompt = super().resolve_prompt(data)
        if not isinstance(prompt, str):
            raise ValueError("Runtime v1 supports a text task prompt, without resumed exchanges")
        return append_guidance(system, self.config.guidance), prompt

    def outbound_messages(self, data):
        system, prompt = self.resolve_prompt(data)
        prefix = BASH_SYSTEM_PROMPT + (" " + EDIT_SYSTEM_PROMPT if self.config.edit else "")
        system = "\n\n".join(part for part in (prefix, system) if part)
        return [{"role": "system", "content": system}, {"role": "user", "content": prompt}]

    async def launch(self, ctx, trace, runtime, endpoint, secret, mcp_urls, data,
                     tool_interception_url=None):
        if mcp_urls:
            raise ValueError("MCP servers are unsupported by this runtime")
        messages = self.outbound_messages(data)
        active_journal().bind(trace, messages)
        request_journal().register(trace_id=trace.id, task_id=trace.task.data.id, model=ctx.model)
        args = ["--bash"] + (["--edit"] if self.config.edit else [])
        if tool_interception_url:
            args.append(f"--tool-interception-url={tool_interception_url}")
        record(trace, "harness_started", guidance_sha256=hashlib.sha256(self.config.guidance.encode()).hexdigest())
        terminal = {"outcome": "exception", "exception_type": None, "exit_code": None,
                    "runtime_alive": None}
        try:
            result = await launch_chat_program(
                chat_program_source(), self.config, ctx, trace, runtime, endpoint, secret, {},
                messages[0]["content"], messages[1]["content"], extra_args=args, env={}, activate=False)
            terminal.update(outcome="returned", exit_code=result.exit_code,
                            stdout_bytes=len(result.stdout.encode()), stderr_bytes=len(result.stderr.encode()),
                            stdout_sha256=hashlib.sha256(result.stdout.encode()).hexdigest(),
                            stderr_sha256=hashlib.sha256(result.stderr.encode()).hexdigest())
            return result
        except BaseException as error:
            terminal.update(outcome="cancelled" if isinstance(error, asyncio.CancelledError) else "exception",
                            exception_type=type(error).__name__)
            raise
        finally:
            # Persist interruption before probing a potentially unresponsive runtime.
            record(trace, "harness_terminal", **terminal)
            try:
                alive = await asyncio.wait_for(runtime.alive(), timeout=2)
            except asyncio.CancelledError:
                record(trace, "runtime_health", alive=None, cancelled=True)
                raise
            except Exception:
                alive = None
            record(trace, "runtime_health", alive=alive)
