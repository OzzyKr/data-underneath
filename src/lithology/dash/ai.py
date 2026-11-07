"""AI providers behind one switch, so the Insights page works with whichever API key the team has.

Set in ``.env`` (repo root) or the environment:

    AI_PROVIDER=anthropic | openai | none      # default: the first provider that is configured, else none
    ANTHROPIC_API_KEY=...   ANTHROPIC_MODEL=claude-opus-5-5   AI_EFFORT=medium
    OPENAI_API_KEY=...      OPENAI_MODEL=<model>   OPENAI_BASE_URL=<any OpenAI-compatible endpoint>

``openai`` covers every OpenAI-compatible endpoint (OpenAI, Azure, Gemini, Groq, Mistral, a local Ollama ...).
``none`` runs offline: insight notes and answers come from the cache of earlier runs.
The provider can also be switched at runtime from the dashboard's top bar.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

ANTHROPIC_DEFAULT_MODEL = "claude-opus-5-5"

ToolSpec = dict[str, Any]  # {"name", "description", "parameters": JSON schema}
Executor = Callable[[str, dict[str, Any]], tuple[str, bool, bool]]  # (result text, is_error, stop_loop)


class AIError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProviderInfo:
    name: str
    label: str
    available: bool
    model: str | None
    reason: str = ""


def provider_infos() -> list[ProviderInfo]:
    a_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")
    a_model = os.getenv("ANTHROPIC_MODEL") or ANTHROPIC_DEFAULT_MODEL
    o_key = os.getenv("OPENAI_API_KEY") or os.getenv("AI_API_KEY")
    o_model = os.getenv("OPENAI_MODEL") or os.getenv("AI_MODEL")
    o_base = os.getenv("OPENAI_BASE_URL")
    o_ok = bool(o_model and (o_key or o_base))
    o_reason = "" if o_ok else ("set OPENAI_MODEL" if (o_key or o_base) else "set OPENAI_API_KEY and OPENAI_MODEL")
    return [
        ProviderInfo("anthropic", "Claude", bool(a_key), a_model, "" if a_key else "set ANTHROPIC_API_KEY"),
        ProviderInfo("openai", "OpenAI-compatible", o_ok, o_model, o_reason),
        ProviderInfo("none", "Offline (cached)", True, None),
    ]


def default_provider() -> str:
    want = (os.getenv("AI_PROVIDER") or "").strip().lower()
    infos = {p.name: p for p in provider_infos()}
    if want in infos and infos[want].available:
        return want
    for p in infos.values():
        if p.available and p.name != "none":
            return p.name
    return "none"


class Provider:
    name = "none"
    model: str | None = None

    def complete(self, system: str, user: str) -> str:
        raise AIError("AI is offline")

    def tool_loop(self, system: str, user: str, tools: list[ToolSpec], execute: Executor, max_steps: int = 8) -> str:
        raise AIError("AI is offline")


class AnthropicProvider(Provider):
    """Claude through the official SDK: manual tool loop, cached system prompt, server-side refusal fallback."""

    name = "anthropic"

    def __init__(self, model: str | None = None, effort: str | None = None):
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = model or os.getenv("ANTHROPIC_MODEL") or ANTHROPIC_DEFAULT_MODEL
        self.effort = effort or os.getenv("AI_EFFORT") or "medium"

    def _create(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> Any:
        kw: dict[str, Any] = {}
        if tools:
            kw["tools"] = tools
        try:
            r = self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",  # a declined request is retried server-side on Anthropic's recommended model
                output_config={"effort": self.effort},
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=messages,
                **kw,
            )
        except self._anthropic.AuthenticationError as e:
            raise AIError("Claude rejected the API key (ANTHROPIC_API_KEY)") from e
        except self._anthropic.RateLimitError as e:
            raise AIError("Claude rate limit reached; try again in a minute") from e
        except self._anthropic.APIStatusError as e:
            raise AIError(f"Claude API error {e.status_code}: {e.message}") from e
        except self._anthropic.APIConnectionError as e:
            raise AIError("cannot reach the Claude API (network)") from e
        if r.stop_reason == "refusal":
            raise AIError("the model declined this request")
        return r

    @staticmethod
    def _text(r: Any) -> str:
        return "".join(b.text for b in r.content if b.type == "text").strip()

    def complete(self, system: str, user: str) -> str:
        return self._text(self._create(system, [{"role": "user", "content": user}]))

    def tool_loop(self, system: str, user: str, tools: list[ToolSpec], execute: Executor, max_steps: int = 8) -> str:
        atools = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in tools]
        messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
        for _ in range(max_steps):
            r = self._create(system, messages, atools)
            messages.append({"role": "assistant", "content": r.content})  # full content: keeps thinking blocks intact
            if r.stop_reason == "pause_turn":
                continue
            uses = [b for b in r.content if b.type == "tool_use"]
            if not uses:
                return self._text(r)
            results, stop = [], False
            for u in uses:
                out, is_err, s = execute(u.name, dict(u.input))
                stop = stop or s
                block: dict[str, Any] = {"type": "tool_result", "tool_use_id": u.id, "content": out}
                if is_err:
                    block["is_error"] = True
                results.append(block)
            if stop:
                return self._text(r)
            messages.append({"role": "user", "content": results})
        raise AIError("the model needed too many steps")


class OpenAICompatProvider(Provider):
    """Any OpenAI-compatible chat-completions endpoint with function calling."""

    name = "openai"

    def __init__(self, model: str | None = None):
        import openai
        from openai import OpenAI

        self._openai = openai
        self.model = model or os.getenv("OPENAI_MODEL") or os.getenv("AI_MODEL")
        if not self.model:
            raise AIError("set OPENAI_MODEL")
        key = os.getenv("OPENAI_API_KEY") or os.getenv("AI_API_KEY") or "not-needed"
        self.client = OpenAI(api_key=key, base_url=os.getenv("OPENAI_BASE_URL") or None)

    def _create(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> Any:
        try:
            kw: dict[str, Any] = {"tools": tools} if tools else {}
            return self.client.chat.completions.create(model=self.model, messages=messages, **kw)
        except self._openai.AuthenticationError as e:
            raise AIError("the API rejected the key (OPENAI_API_KEY)") from e
        except self._openai.RateLimitError as e:
            raise AIError("rate limit reached; try again in a minute") from e
        except self._openai.APIStatusError as e:
            raise AIError(f"API error {e.status_code}: {e.message}") from e
        except self._openai.APIConnectionError as e:
            raise AIError("cannot reach the AI endpoint (network or OPENAI_BASE_URL)") from e

    def complete(self, system: str, user: str) -> str:
        r = self._create([{"role": "system", "content": system}, {"role": "user", "content": user}])
        return (r.choices[0].message.content or "").strip()

    def tool_loop(self, system: str, user: str, tools: list[ToolSpec], execute: Executor, max_steps: int = 8) -> str:
        otools = [{"type": "function", "function": t} for t in tools]
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for _ in range(max_steps):
            msg = self._create(messages, otools).choices[0].message
            messages.append(msg.model_dump(exclude_none=True))
            if not msg.tool_calls:
                return (msg.content or "").strip()
            stop = False
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    out, s = "arguments were not valid JSON", False
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})
                    continue
                out, _is_err, s = execute(tc.function.name, args)
                stop = stop or s
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})
            if stop:
                return (msg.content or "").strip()
        raise AIError("the model needed too many steps")


def make_provider(name: str) -> Provider:
    if name == "anthropic":
        return AnthropicProvider()
    if name == "openai":
        return OpenAICompatProvider()
    return Provider()
