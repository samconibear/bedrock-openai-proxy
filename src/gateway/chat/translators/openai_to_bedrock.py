"""Translates an OpenAI Chat Completions request into Bedrock Converse API arguments."""

import json

from src.shared.exceptions import ValidationError
from src.gateway.chat.models import (
    BedrockConverseArgs,
    BedrockInferenceConfig,
    BedrockMessage,
    BedrockSystemBlock,
    BedrockTextBlock,
    BedrockToolConfig,
    BedrockToolResultBlock,
    BedrockToolUseBlock,
)


def translate(openai_request, bedrock_model_id) -> BedrockConverseArgs:
    req = openai_request
    messages = req.messages

    system_parts = []
    for m in messages:
        if m.role == "system":
            if isinstance(m.content, str):
                system_parts.append(m.content)
            elif isinstance(m.content, list):
                system_parts.append(" ".join(p.text for p in m.content if p.type == "text"))
    system = [BedrockSystemBlock(text="\n\n".join(system_parts))] if system_parts else None

    bedrock_messages: list[BedrockMessage] = []
    for m in messages:
        if m.role == "system":
            continue

        if m.role == "tool":
            content_text = m.content if isinstance(m.content, str) else ""
            bedrock_messages.append(BedrockMessage(
                role="user",
                content=[BedrockToolResultBlock(
                    toolUseId=m.tool_call_id,
                    content=[BedrockTextBlock(text=content_text)],
                )],
            ))
            continue

        if m.role == "assistant" and m.tool_calls:
            tool_use_blocks = []
            for tc in m.tool_calls:
                try:
                    input_dict = json.loads(tc.function.arguments)
                except json.JSONDecodeError as e:
                    raise ValidationError(
                        f"tool_calls[{tc.id}].function.arguments is not valid JSON: {e}"
                    )
                tool_use_blocks.append(BedrockToolUseBlock(
                    toolUseId=tc.id,
                    name=tc.function.name,
                    input=input_dict,
                ))
            bedrock_messages.append(BedrockMessage(role="assistant", content=tool_use_blocks))
            continue

        if isinstance(m.content, list):
            _validate_content_parts(m.content)
            text = " ".join(p.text for p in m.content if p.type == "text")
        elif isinstance(m.content, str):
            text = m.content
        else:
            text = ""

        bedrock_messages.append(BedrockMessage(
            role=m.role,
            content=[BedrockTextBlock(text=text)],
        ))

    tool_config = _build_tool_config(req.tools, req.tool_choice)

    inference_config = None
    if req.max_tokens is not None or req.temperature is not None:
        inference_config = BedrockInferenceConfig(
            maxTokens=req.max_tokens,
            temperature=req.temperature,
        )

    return BedrockConverseArgs(
        model_id=bedrock_model_id,
        messages=bedrock_messages,
        system=system,
        tools=tool_config,
        inference_config=inference_config,
    )


def _validate_content_parts(parts) -> None:
    for part in parts:
        if part.type != "text":
            raise ValidationError(
                "Image and document content is not supported in Phase 1. "
                "Only text content is accepted."
            )


def _build_tool_config(tools, tool_choice) -> BedrockToolConfig | None:
    if tool_choice == "none":
        return None  # Bedrock has no "ignore tools" mode

    if not tools:
        return None

    bedrock_tools = []
    for t in tools:
        fn = t.function
        tool_spec = {
            "name": fn.name,
            "inputSchema": {"json": fn.parameters or {}},
        }
        if fn.description:
            tool_spec["description"] = fn.description
        bedrock_tools.append({"toolSpec": tool_spec})

    bedrock_choice = _translate_tool_choice(tool_choice)
    return BedrockToolConfig(tools=bedrock_tools, toolChoice=bedrock_choice)


def _translate_tool_choice(tool_choice) -> dict | None:
    if tool_choice is None or tool_choice == "auto":
        return {"auto": {}}
    if tool_choice == "required":
        return {"any": {}}
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        name = tool_choice.get("function", {}).get("name")
        return {"tool": {"name": name}}
    return None
