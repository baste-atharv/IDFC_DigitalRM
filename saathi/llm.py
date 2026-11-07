"""One interface over several LLM providers, with tool calling.

- Anthropic (Claude) via the `anthropic` SDK (Messages API, tool_use / tool_result blocks)
- Everything else via the `openai` SDK's chat.completions with a base_url:
  OpenAI, Google Gemini (OpenAI-compatible endpoint), Groq, OpenRouter, or a local Ollama
  model (an open model such as Llama 3 on Ollama keeps data in-house).
"""
from __future__ import annotations

import json

from . import config as C


class LLMError(RuntimeError):
    pass


class LLM:
    def __init__(self, provider: str, api_key: str = "", model: str = "", base_url: str = ""):
        if provider not in C.PROVIDERS:
            raise LLMError(f"Unknown provider {provider}")
        p = C.PROVIDERS[provider]
        self.provider, self.kind = provider, p["kind"]
        self.model = model or p["default_model"]
        self.base_url = base_url or p["base_url"]
        self.api_key = api_key
        self.client = None
        if self.kind == "offline":
            return
        if self.kind == "anthropic":
            try:
                import anthropic
            except ImportError as e:
                raise LLMError("Package 'anthropic' missing: run SETUP_AND_RUN.bat again") from e
            if not api_key:
                raise LLMError("Anthropic API key is empty")
            self.client = anthropic.Anthropic(api_key=api_key)
        else:
            try:
                import openai
            except ImportError as e:
                raise LLMError("Package 'openai' missing: run SETUP_AND_RUN.bat again") from e
            if not api_key and "localhost" not in (self.base_url or ""):
                raise LLMError(f"{provider} API key is empty")
            self.client = openai.OpenAI(api_key=api_key or "ollama", base_url=self.base_url)

    @property
    def online(self) -> bool:
        return self.kind != "offline"

    # ------------------------------------------------------------------
    def list_models(self) -> list[str]:
        if self.kind == "anthropic":
            return sorted(m.id for m in self.client.models.list(limit=100).data)
        if self.kind == "openai":
            return sorted(m.id for m in self.client.models.list().data)
        return ["rules"]

    def complete(self, prompt: str, system: str = "", max_tokens: int = 400) -> str:
        msgs = [{"role": "user", "content": prompt}]
        text, _ = self.chat(msgs, system=system, tools=None, max_tokens=max_tokens)
        return text

    def chat(self, messages: list[dict], system: str, tools: dict | None, executor=None,
             max_tokens: int = 700, max_steps: int = 5) -> tuple[str, list[dict]]:
        """Runs the tool loop. `tools` = {name: (description, json_schema)}; `executor(name, args) -> dict`.
        Returns final text and a trace of tool calls."""
        if self.kind == "anthropic":
            return self._chat_anthropic(messages, system, tools, executor, max_tokens, max_steps)
        if self.kind == "openai":
            return self._chat_openai(messages, system, tools, executor, max_tokens, max_steps)
        raise LLMError("Offline mode has no LLM")

    # ------------------------------------------------------------------ Anthropic
    def _chat_anthropic(self, messages, system, tools, executor, max_tokens, max_steps):
        msgs = [dict(m) for m in messages]
        tdefs = [dict(name=n, description=d, input_schema=s) for n, (d, s) in (tools or {}).items()]
        trace = []
        for _ in range(max_steps):
            kw = dict(model=self.model, max_tokens=max_tokens, system=system, messages=msgs)
            if tdefs:
                kw["tools"] = tdefs
            try:
                r = self.client.messages.create(**kw)
            except Exception as e:
                raise LLMError(f"Anthropic call failed: {e}") from e
            blocks, calls = [], []
            for b in r.content:
                if b.type == "text":
                    blocks.append(dict(type="text", text=b.text))
                elif b.type == "tool_use":
                    blocks.append(dict(type="tool_use", id=b.id, name=b.name, input=b.input))
                    calls.append(b)
            if r.stop_reason != "tool_use" or not calls or executor is None:
                return "".join(b["text"] for b in blocks if b["type"] == "text").strip(), trace
            msgs.append(dict(role="assistant", content=blocks))
            results = []
            for c in calls:
                out = executor(c.name, dict(c.input or {}))
                trace.append(dict(tool=c.name, args=c.input, result=out))
                results.append(dict(type="tool_result", tool_use_id=c.id, content=json.dumps(out, ensure_ascii=False, default=str)))
            msgs.append(dict(role="user", content=results))
        return "Sorry, I could not finish that. Let me connect you to your RM.", trace

    # ------------------------------------------------------------------ OpenAI-compatible
    def _chat_openai(self, messages, system, tools, executor, max_tokens, max_steps):
        msgs = [dict(role="system", content=system)] + [dict(m) for m in messages]
        tdefs = [dict(type="function", function=dict(name=n, description=d, parameters=s)) for n, (d, s) in (tools or {}).items()]
        trace = []
        for _ in range(max_steps):
            kw = dict(model=self.model, messages=msgs, max_tokens=max_tokens)
            if tdefs:
                kw["tools"] = tdefs
            try:
                r = self.client.chat.completions.create(**kw)
            except Exception as e:
                if "max_completion_tokens" in str(e):          # newer OpenAI models renamed the limit
                    kw["max_completion_tokens"] = kw.pop("max_tokens")
                    try:
                        r = self.client.chat.completions.create(**kw)
                    except Exception as e2:
                        raise LLMError(f"{self.provider} call failed: {e2}") from e2
                else:
                    raise LLMError(f"{self.provider} call failed: {e}") from e
            m = r.choices[0].message
            calls = getattr(m, "tool_calls", None) or []
            if not calls or executor is None:
                return (m.content or "").strip(), trace
            msgs.append(dict(role="assistant", content=m.content or "",
                             tool_calls=[dict(id=c.id, type="function",
                                              function=dict(name=c.function.name, arguments=c.function.arguments or "{}"))
                                         for c in calls]))
            for c in calls:
                try:
                    args = json.loads(c.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                out = executor(c.function.name, args)
                trace.append(dict(tool=c.function.name, args=args, result=out))
                msgs.append(dict(role="tool", tool_call_id=c.id, content=json.dumps(out, ensure_ascii=False, default=str)))
        return "Sorry, I could not finish that. Let me connect you to your RM.", trace


def test_connection(llm: LLM) -> str:
    if not llm.online:
        return "Offline mode: no LLM needed."
    txt = llm.complete("Reply with exactly: OK", max_tokens=10)
    return f"Connected to {llm.provider} / {llm.model}: replied '{txt[:30]}'"
