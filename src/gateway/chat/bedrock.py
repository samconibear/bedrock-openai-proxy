import boto3
from botocore.exceptions import ClientError

from src.shared.exceptions import ProviderError, RateLimitError
from src.gateway.chat.models import (
    BedrockConverseArgs,
    BedrockResponse,
    BedrockTextBlock,
    BedrockToolUseBlock,
    BedrockUsage,
)

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = boto3.client("bedrock-runtime")
    return _client


def _build_converse_kwargs(args: BedrockConverseArgs) -> dict:
    kwargs: dict = {
        "modelId": args.model_id,
        "messages": [
            {
                "role": m.role,
                "content": [_serialise_block(b) for b in m.content],
            }
            for m in args.messages
        ],
    }
    if args.system:
        kwargs["system"] = [{"text": s.text} for s in args.system]
    if args.tools:
        tc: dict = {"tools": args.tools.tools}
        if args.tools.toolChoice is not None:
            tc["toolChoice"] = args.tools.toolChoice
        kwargs["toolConfig"] = tc
    if args.inference_config:
        ic = {}
        if args.inference_config.maxTokens is not None:
            ic["maxTokens"] = args.inference_config.maxTokens
        if args.inference_config.temperature is not None:
            ic["temperature"] = args.inference_config.temperature
        if ic:
            kwargs["inferenceConfig"] = ic
    return kwargs


def _serialise_block(block) -> dict:
    if isinstance(block, BedrockTextBlock):
        return {"text": block.text}
    if isinstance(block, BedrockToolUseBlock):
        return {"toolUse": {"toolUseId": block.toolUseId, "name": block.name, "input": block.input}}
    return {
        "toolResult": {
            "toolUseId": block.toolUseId,
            "content": [{"text": c.text} for c in block.content],
        }
    }


def invoke(args: BedrockConverseArgs) -> BedrockResponse:
    client = _get_client()
    kwargs = _build_converse_kwargs(args)
    try:
        resp = client.converse(**kwargs)
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code == "ThrottlingException":
            raise RateLimitError(str(e)) from e
        raise ProviderError(str(e)) from e
    except Exception as e:
        raise ProviderError(str(e)) from e

    output_message = resp["output"]["message"]
    content_blocks = []
    for block in output_message.get("content", []):
        if "text" in block:
            content_blocks.append(BedrockTextBlock(text=block["text"]))
        elif "toolUse" in block:
            tu = block["toolUse"]
            content_blocks.append(BedrockToolUseBlock(
                toolUseId=tu["toolUseId"],
                name=tu["name"],
                input=tu["input"],
            ))

    usage_raw = resp.get("usage", {})
    return BedrockResponse(
        content=content_blocks,
        stop_reason=resp["stopReason"],
        usage=BedrockUsage(
            inputTokens=usage_raw.get("inputTokens", 0),
            outputTokens=usage_raw.get("outputTokens", 0),
        ),
    )


def stream(args: BedrockConverseArgs):
    client = _get_client()
    kwargs = _build_converse_kwargs(args)
    try:
        resp = client.converse_stream(**kwargs)
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code == "ThrottlingException":
            raise RateLimitError(str(e)) from e
        raise ProviderError(str(e)) from e
    except Exception as e:
        raise ProviderError(str(e)) from e

    return resp["stream"]
