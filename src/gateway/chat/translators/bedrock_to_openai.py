"""Translates a Bedrock Converse response into an OpenAI Chat Completions response."""

import json
import time
import uuid

from src.gateway.chat.models import BedrockResponse, BedrockTextBlock, BedrockToolUseBlock

_STOP_REASON_MAP = {
    "end_turn": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
}


def translate(resp: BedrockResponse, model: str) -> dict:
    message: dict = {"role": "assistant"}

    text_parts = [b.text for b in resp.content if isinstance(b, BedrockTextBlock)]
    tool_use_blocks = [b for b in resp.content if isinstance(b, BedrockToolUseBlock)]

    message["content"] = "\n".join(text_parts) if text_parts else None

    if tool_use_blocks:
        message["tool_calls"] = [
            {
                "id": b.toolUseId,
                "type": "function",
                "function": {
                    "name": b.name,
                    "arguments": json.dumps(b.input),
                },
            }
            for b in tool_use_blocks
        ]

    finish_reason = _STOP_REASON_MAP.get(resp.stop_reason, resp.stop_reason)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": resp.usage.inputTokens,
            "completion_tokens": resp.usage.outputTokens,
            "total_tokens": resp.usage.inputTokens + resp.usage.outputTokens,
        },
    }
