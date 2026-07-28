"""
LLM client factory.

OpenAI path exposes an Anthropic-compatible ``client.messages.create`` surface,
including tool-use / tool_result message plumbing used by ``run_agent``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4
import json
import os


@dataclass
class _TextBlock:
    type: str = "text"
    text: str = ""


@dataclass
class _ToolUseBlock:
    type: str = "tool_use"
    id: str = ""
    name: str = ""
    input: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Response:
    content: list[Any]
    stop_reason: str = "end_turn"


def _anthropic_tools_to_openai(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    converted = []
    for tool in tools or []:
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema")
                    or {"type": "object", "properties": {}},
                },
            }
        )
    return converted


def _flatten_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif block.get("type") == "tool_result":
                    # Handled separately as role=tool messages.
                    continue
                else:
                    parts.append(json.dumps(block, default=str))
            elif hasattr(block, "type") and getattr(block, "type", None) == "text":
                parts.append(getattr(block, "text", "") or "")
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


class _MessagesAPI:
    def __init__(self, openai_client: Any):
        self._client = openai_client

    def create(
        self,
        *,
        model: str,
        max_tokens: int = 2000,
        temperature: float = 0,
        system: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        **_: Any,
    ) -> _Response:
        oai_messages: list[dict[str, Any]] = []
        if system:
            oai_messages.append({"role": "system", "content": system})

        for msg in messages or []:
            role = msg.get("role", "user")
            content = msg.get("content")

            # Anthropic assistant turn with mixed text + tool_use blocks
            if role == "assistant" and isinstance(content, list):
                text_parts: list[str] = []
                tool_calls = []
                for block in content:
                    btype = getattr(block, "type", None)
                    if btype is None and isinstance(block, dict):
                        btype = block.get("type")
                    if btype == "text":
                        text_parts.append(
                            getattr(block, "text", None)
                            if not isinstance(block, dict)
                            else block.get("text", "")
                        )
                    elif btype == "tool_use":
                        if isinstance(block, dict):
                            tid = block.get("id") or f"tool_{uuid4().hex[:8]}"
                            name = block.get("name", "")
                            args = block.get("input", {})
                        else:
                            tid = getattr(block, "id", f"tool_{uuid4().hex[:8]}")
                            name = getattr(block, "name", "")
                            args = getattr(block, "input", {}) or {}
                        tool_calls.append(
                            {
                                "id": tid,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(args),
                                },
                            }
                        )
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": "".join(text_parts) or None,
                }
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
                oai_messages.append(assistant_msg)
                continue

            # Anthropic user turn carrying tool_result blocks
            if role == "user" and isinstance(content, list):
                has_tool_result = any(
                    (isinstance(b, dict) and b.get("type") == "tool_result")
                    or getattr(b, "type", None) == "tool_result"
                    for b in content
                )
                if has_tool_result:
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_result":
                            oai_messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": block.get("tool_use_id", ""),
                                    "content": str(block.get("content", "")),
                                }
                            )
                        elif getattr(block, "type", None) == "tool_result":
                            oai_messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": getattr(block, "tool_use_id", ""),
                                    "content": str(getattr(block, "content", "")),
                                }
                            )
                    continue

            oai_messages.append(
                {"role": role, "content": _flatten_content(content)}
            )

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": oai_messages,
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
        }
        oai_tools = _anthropic_tools_to_openai(tools)
        if oai_tools:
            kwargs["tools"] = oai_tools

        completion = self._client.chat.completions.create(**kwargs)
        message = completion.choices[0].message

        blocks: list[Any] = []
        if message.content:
            blocks.append(_TextBlock(type="text", text=message.content))

        tool_calls = message.tool_calls or []
        for call in tool_calls:
            raw_args = call.function.arguments or "{}"
            try:
                parsed = json.loads(raw_args)
            except json.JSONDecodeError:
                parsed = {"_raw": raw_args}
            if not isinstance(parsed, dict):
                parsed = {"value": parsed}
            blocks.append(
                _ToolUseBlock(
                    type="tool_use",
                    id=call.id,
                    name=call.function.name,
                    input=parsed,
                )
            )

        stop_reason = "tool_use" if tool_calls else "end_turn"
        return _Response(content=blocks, stop_reason=stop_reason)


class OpenAIAnthropicCompatClient:
    """Drop-in for code that expects client.messages.create(...)."""

    def __init__(self, api_key: str | None = None):
        from openai import OpenAI

        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self._openai = OpenAI(api_key=key)
        self.messages = _MessagesAPI(self._openai)


def build_llm_client() -> tuple[Any, str]:
    """
    Prefer OpenAI when OPENAI_API_KEY is set; else Anthropic.
    Returns (client, default_model).
    """
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if openai_key and not openai_key.endswith("here"):
        model = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
        return OpenAIAnthropicCompatClient(api_key=openai_key), model

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if anthropic_key and "your-key" not in anthropic_key:
        import anthropic

        model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        return anthropic.Anthropic(api_key=anthropic_key), model

    raise RuntimeError(
        "No usable API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env."
    )
