"""
Translates Bedrock ConverseStream events into OpenAI-compatible SSE chunks.

Tool call blocks are buffered until contentBlockStop, then emitted whole.
Token counts arrive in the metadata event and are stored for the caller to log.
"""

import json
import time
import uuid

from src.shared.exceptions import ProviderError

_STOP_REASON_MAP = {
    "end_turn": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
}


class SSETranslator:
    def __init__(self, model_alias: str):
        self._model_alias = model_alias
        self._completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        self._created = int(time.time())
        self.input_tokens: int = 0  # readable by handler after iteration
        self.output_tokens: int = 0

    def translate(self, event_stream) -> "Generator[str, None, None]":
        _current_tool: dict | None = None
        _finish_reason: str | None = None
        _usage_chunk: str | None = None

        try:
            for event in event_stream:
                if "messageStart" in event:
                    continue

                if "contentBlockStart" in event:
                    start = event["contentBlockStart"].get("start", {})
                    if "toolUse" in start:
                        _current_tool = {
                            "id": start["toolUse"]["toolUseId"],
                            "name": start["toolUse"]["name"],
                            "arguments": "",
                        }
                    continue

                if "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"].get("delta", {})
                    if "text" in delta:
                        yield self._text_chunk(delta["text"])
                    elif "toolUse" in delta and _current_tool is not None:
                        _current_tool["arguments"] += delta["toolUse"].get("input", "")
                    continue

                if "contentBlockStop" in event:
                    if _current_tool is not None:
                        yield self._tool_call_chunk(_current_tool)
                        _current_tool = None
                    continue

                if "messageStop" in event:
                    stop_reason = event["messageStop"].get("stopReason", "end_turn")
                    _finish_reason = _STOP_REASON_MAP.get(stop_reason, stop_reason)
                    continue

                if "metadata" in event:
                    usage = event["metadata"].get("usage", {})
                    self.input_tokens = usage.get("inputTokens", 0)
                    self.output_tokens = usage.get("outputTokens", 0)
                    _usage_chunk = self._usage_chunk()
                    continue

                if "internalServerException" in event or "modelStreamErrorException" in event:
                    raise ProviderError(f"Bedrock stream error: {event}")

        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(f"Stream processing error: {e}") from e

        yield self._final_chunk(_finish_reason or "stop")

        if _usage_chunk:
            yield _usage_chunk

        yield "data: [DONE]\n\n"

    def _base_chunk(self) -> dict:
        return {
            "id": self._completion_id,
            "object": "chat.completion.chunk",
            "created": self._created,
            "model": self._model_alias,
            "choices": [],
        }

    def _text_chunk(self, text: str) -> str:
        chunk = self._base_chunk()
        chunk["choices"] = [{"index": 0, "delta": {"content": text}, "finish_reason": None}]
        return f"data: {json.dumps(chunk)}\n\n"

    def _tool_call_chunk(self, tool: dict) -> str:
        chunk = self._base_chunk()
        chunk["choices"] = [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": tool["id"],
                            "type": "function",
                            "function": {"name": tool["name"], "arguments": tool["arguments"]},
                        }
                    ]
                },
                "finish_reason": None,
            }
        ]
        return f"data: {json.dumps(chunk)}\n\n"

    def _final_chunk(self, finish_reason: str) -> str:
        chunk = self._base_chunk()
        chunk["choices"] = [{"index": 0, "delta": {}, "finish_reason": finish_reason}]
        return f"data: {json.dumps(chunk)}\n\n"

    def _usage_chunk(self) -> str:
        chunk = self._base_chunk()
        chunk["choices"] = []
        chunk["usage"] = {
            "prompt_tokens": self.input_tokens,
            "completion_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
        }
        return f"data: {json.dumps(chunk)}\n\n"
