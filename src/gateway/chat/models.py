"""Typed boundary objects for the chat completions feature."""

from __future__ import annotations
from pydantic import BaseModel

from src.shared.models import UserContext


class ToolFunction(BaseModel):
    name: str
    description: str | None = None
    parameters: dict | None = None


class Tool(BaseModel):
    type: str  # always "function"
    function: ToolFunction


class MessageContentPart(BaseModel):
    type: str
    text: str


class ToolCallFunction(BaseModel):
    name: str
    arguments: str  # JSON-encoded per OpenAI spec


class ToolCall(BaseModel):
    id: str
    type: str  # always "function"
    function: ToolCallFunction


class Message(BaseModel):
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str | list[MessageContentPart] | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None  # present when role="tool"


class OpenAIRequest(BaseModel):
    model: str
    messages: list[Message]
    tools: list[Tool] | None = None
    tool_choice: str | dict | None = None
    stream: bool = False
    max_tokens: int | None = None
    temperature: float | None = None



class BedrockTextBlock(BaseModel):
    type: str = "text"
    text: str


class BedrockToolUseBlock(BaseModel):
    type: str = "toolUse"
    toolUseId: str
    name: str
    input: dict  # parsed JSON; serialised back to arguments string for OpenAI


class BedrockToolResultBlock(BaseModel):
    type: str = "toolResult"
    toolUseId: str
    content: list[BedrockTextBlock]


class BedrockMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: list[BedrockTextBlock | BedrockToolUseBlock | BedrockToolResultBlock]


class BedrockSystemBlock(BaseModel):
    text: str


class BedrockToolSpec(BaseModel):
    name: str
    description: str | None = None
    inputSchema: dict  # {"json": <JSON Schema dict>}


class BedrockToolConfig(BaseModel):
    tools: list[dict]       # [{"toolSpec": BedrockToolSpec.model_dump()}]
    toolChoice: dict | None = None


class BedrockInferenceConfig(BaseModel):
    maxTokens: int | None = None
    temperature: float | None = None


class BedrockConverseArgs(BaseModel):
    model_id: str
    messages: list[BedrockMessage]
    system: list[BedrockSystemBlock] | None = None
    tools: BedrockToolConfig | None = None
    inference_config: BedrockInferenceConfig | None = None


class BedrockUsage(BaseModel):
    inputTokens: int
    outputTokens: int


class BedrockResponse(BaseModel):
    content: list[BedrockTextBlock | BedrockToolUseBlock]
    stop_reason: str
    usage: BedrockUsage
