"""Pinned native training harness with exact host prompt binding and zero retries."""

import asyncio
import hashlib
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harness import Harness
from verifiers.v1.harnesses.bash.harness import BASH_SYSTEM_PROMPT, EDIT_SYSTEM_PROMPT
from verifiers.v1.harnesses.utils import compaction, mcp
from verifiers.v1.harnesses.utils.launch import CHAT_PROGRAM, bundle_program, launch_chat_program
from verifiers.v1.utils import network

from . import _chat_core
from .journal import owner, record

DEPENDENCIES = {"openai": "2.54.0", "mcp": "2.3.0", "httpx": "0.28.1",
                "httpx2": "2.16.0", "tenacity": "9.2.1", "certifi": "2026.10.23"}


def chat_program_source():
    header = '# dependencies = ["openai", "mcp==2.3.0", "httpx", "httpx2", "tenacity", "certifi"]'
    if CHAT_PROGRAM.count(header) != 1:
        raise RuntimeError("Unexpected pinned native chat dependency header")
    pinned = ", ".join(f'"{name}=={version}"' for name, version in DEPENDENCIES.items())
    template = CHAT_PROGRAM.replace(header, f"# dependencies = [{pinned}]")
    notice = Path(__file__).with_name("THIRD_PARTY_NOTICES.txt").read_text()
    return bundle_program(template + "\n".join("# " + line for line in notice.splitlines()) + "\n",
                          network, mcp, compaction, _chat_core)


class FinancialSelectionTrainingHarnessConfig(HarnessConfig):
    id: str = "financial-selection-training"
    dialect: Literal["chat_completions"] = "chat_completions"
    edit: bool = True
    search: Literal[False] = False
    compaction: None = None
    sdk_max_retries: Literal[0] = 0

    @field_validator("sdk_max_retries", mode="before")
    @classmethod
    def no_retries(cls, value):
        if type(value) is not int or value != 0:
            raise ValueError("Transport retries must be integer zero")
        return value

    @field_validator("env", "forward_env", "mcp_header_env", "skills")
    @classmethod
    def no_external_context(cls, value):
        if value:
            raise ValueError("Training harness external context must be empty")
        return value

    @field_validator("disabled_tools")
    @classmethod
    def no_tool_override(cls, value):
        if value is not None:
            raise ValueError("Training harness tool overrides are unsupported")
        return value


class FinancialSelectionTrainingHarness(Harness[FinancialSelectionTrainingHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_RESUME = False
    SUPPORTS_TOOL_INTERCEPTION = True
    NEEDS_CONTAINER = True

    async def setup(self, runtime):
        await runtime.prepare_uv_script(chat_program_source(), self.config.resolved_env)

    def outbound_messages(self, data):
        system, prompt = self.resolve_prompt(data)
        if not isinstance(prompt, str):
            raise ValueError("Training v1 requires a text task prompt")
        prefix = BASH_SYSTEM_PROMPT + (" " + EDIT_SYSTEM_PROMPT if self.config.edit else "")
        return [{"role": "system", "content": "\n\n".join(p for p in (prefix, system) if p)},
                {"role": "user", "content": prompt}]

    async def launch(self, ctx, trace, runtime, endpoint, secret, mcp_urls, data,
                     tool_interception_url=None):
        if mcp_urls:
            raise ValueError("Training v1 does not allow external MCP servers")
        journal, _ = owner(trace.id)
        messages = self.outbound_messages(data)
        journal.bind(trace, messages)
        record(trace, "harness_started")
        terminal = {"outcome": "exception", "exit_code": None, "exception_type": None}
        try:
            args = ["--bash"] + (["--edit"] if self.config.edit else [])
            if tool_interception_url:
                args.append(f"--tool-interception-url={tool_interception_url}")
            result = await launch_chat_program(chat_program_source(), self.config, ctx, trace, runtime,
                                               endpoint, secret, {}, messages[0]["content"], messages[1]["content"],
                                               extra_args=args, env={}, activate=False)
            terminal.update(outcome="returned", exit_code=result.exit_code,
                            stdout_sha256=hashlib.sha256(result.stdout.encode()).hexdigest(),
                            stderr_sha256=hashlib.sha256(result.stderr.encode()).hexdigest())
            return result
        except BaseException as error:
            terminal.update(outcome="cancelled" if isinstance(error, asyncio.CancelledError) else "exception",
                            exception_type=type(error).__name__)
            raise
        finally:
            record(trace, "harness_terminal", **terminal)
            try:
                alive = await asyncio.wait_for(runtime.alive(), timeout=2)
            except asyncio.CancelledError:
                record(trace, "runtime_health", alive=None, cancelled=True)
                raise
            except Exception:
                alive = None
            record(trace, "runtime_health", alive=alive)
