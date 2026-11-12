"""Restricted native chat harness with explicitly disabled transport retries."""

from importlib.metadata import version
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harness import Harness
from verifiers.v1.harnesses.bash.harness import BASH_SYSTEM_PROMPT, EDIT_SYSTEM_PROMPT
from verifiers.v1.harnesses.utils import compaction, mcp
from verifiers.v1.harnesses.utils.launch import (
    CHAT_PROGRAM,
    bundle_program,
    launch_chat_program,
)
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace
from verifiers.v1.utils import network

from . import _chat_core


_UPSTREAM_DEPENDENCIES = (
    '# dependencies = ["openai", "mcp==2.3.0", "httpx", "httpx2", "tenacity", "certifi"]'
)
BUNDLED_DEPENDENCIES = {
    "openai": "2.54.0",
    "mcp": "2.3.0",
    "httpx": "0.28.1",
    "httpx2": "2.16.0",
    "tenacity": "9.2.1",
    "certifi": "2026.10.23",
}


def chat_program_source() -> str:
    """Bundle the pinned native helpers and the narrowly modified chat core."""
    if version("verifiers") != "0.4.1":
        raise RuntimeError("financial-portfolio-staged requires verifiers==0.4.1")
    if CHAT_PROGRAM.count(_UPSTREAM_DEPENDENCIES) != 1:
        raise RuntimeError("unexpected native chat dependency header")
    pinned = ", ".join(f'"{name}=={release}"' for name, release in BUNDLED_DEPENDENCIES.items())
    template = CHAT_PROGRAM.replace(_UPSTREAM_DEPENDENCIES, f"# dependencies = [{pinned}]")
    notice = Path(__file__).with_name("THIRD_PARTY_LICENSES.txt").read_text()
    metadata = template + "\n".join(f"# {line}" for line in notice.splitlines()) + "\n"
    return bundle_program(metadata, network, mcp, compaction, _chat_core)


class FinancialPortfolioStagedHarnessConfig(HarnessConfig):
    id: str = "financial-portfolio-staged"
    dialect: Literal["chat_completions"] = "chat_completions"
    edit: bool = True
    search: Literal[False] = False
    compaction: None = None
    sdk_max_retries: Literal[0] = 0

    @field_validator("sdk_max_retries", mode="before")
    @classmethod
    def zero_retries(cls, value):
        if type(value) is not int or value != 0:
            raise ValueError("sdk_max_retries must be the integer 0")
        return value

    @field_validator("env", "forward_env", "mcp_header_env", "skills")
    @classmethod
    def no_external_context(cls, value):
        if value:
            raise ValueError("the staged harness requires this setting to be empty")
        return value

    @field_validator("disabled_tools")
    @classmethod
    def no_tool_overrides(cls, value):
        if value is not None:
            raise ValueError("disabled_tools is unsupported by the staged harness")
        return value


class FinancialPortfolioStagedHarness(Harness[FinancialPortfolioStagedHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_RESUME = True
    SUPPORTS_TOOL_INTERCEPTION = True
    NEEDS_CONTAINER = True

    async def setup(self, runtime: Runtime) -> None:
        await runtime.prepare_uv_script(chat_program_source(), self.config.resolved_env)

    async def launch(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: TaskData,
        tool_interception_url: str | None = None,
    ) -> ProgramResult:
        if mcp_urls:
            raise ValueError("MCP servers are unsupported by the staged harness")
        system_prompt, prompt = self.resolve_prompt(data)
        harness_prompt = BASH_SYSTEM_PROMPT
        if self.config.edit:
            harness_prompt += " " + EDIT_SYSTEM_PROMPT
        system_prompt = "\n\n".join(p for p in (harness_prompt, system_prompt) if p)
        if (
            data.system_prompt is None
            and prompt is not None
            and not isinstance(prompt, str)
            and any(
                message.role == "system" and message.content == harness_prompt
                for message in prompt
            )
        ):
            system_prompt = None
        args = ["--bash"]
        if self.config.edit:
            args.append("--edit")
        if tool_interception_url:
            args.append(f"--tool-interception-url={tool_interception_url}")
        return await launch_chat_program(
            chat_program_source(),
            self.config,
            ctx,
            trace,
            runtime,
            endpoint,
            secret,
            {},
            system_prompt,
            prompt,
            extra_args=args,
            env={},
            activate=False,
        )
