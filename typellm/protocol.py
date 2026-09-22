"""Chat-template protocol differences, independent of the serving backend."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ChatProtocol:
    name: str = "think-tags"
    thinking_open: str = "<think>"
    thinking_close: str = "</think>"
    answer_separator: str = "\n\n"
    turn_end: str | None = None

    def has_open_thinking(self, prefix: str) -> bool:
        return prefix.rstrip().endswith(self.thinking_open)

    def forced_close(self) -> str:
        return "\n\nI will now give the final answer.\n" + self.thinking_close + self.answer_separator

    def answer_prefix(self, prefix: str, reasoning: str) -> str:
        return prefix + reasoning + self.thinking_close + self.answer_separator


def detect_protocol(tokenizer: Any) -> ChatProtocol:
    """Detect protocol from template control tokens, not a model-name guess."""
    template = getattr(tokenizer, "chat_template", "") or ""
    if not isinstance(template, str):
        template = tokenizer.get_chat_template()
    if "<|turn>" in template and "<|channel>" in template:
        raise ValueError("Gemma 4 chat protocol is not supported")
    if "<|im_end|>" in template:
        return ChatProtocol(name="chatml", turn_end="<|im_end|>")
    if "<|role_end|>" in template:
        return ChatProtocol(name="ling", turn_end="<|role_end|>")
    if "<role>" in template:
        return ChatProtocol(name="ring")
    return ChatProtocol()
