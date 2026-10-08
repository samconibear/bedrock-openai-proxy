import json
import time

from aws_lambda_powertools import Logger

from src.shared.exceptions import ValidationError
from src.shared.models import UserContext
from src.gateway.chat.models import OpenAIRequest
from src.gateway.chat.translators import (
  to_bedrock,
  to_openai,
  SSETranslator,
)
from src.gateway.chat.bedrock import invoke as bedrock_invoke, stream as bedrock_stream
from src.gateway.shared.model_map import MODEL_MAP
logger = Logger()


def handle(event: dict, user_context: UserContext):
    raw_body = event.get("body", "") or ""
    try:
        body = json.loads(raw_body)
    except json.JSONDecodeError as e:
        raise ValidationError(f"Request body is not valid JSON: {e}") from e

    try:
        openai_req = OpenAIRequest.model_validate(body)
    except Exception as e:
        raise ValidationError(f"Invalid request body: {e}") from e

    _validate_content(openai_req)

    if openai_req.model not in MODEL_MAP:
        raise Exception(f"Unknown model: {openai_req.model}. Available aliases: {list(MODEL_MAP)}")
    bedrock_model_id = MODEL_MAP[openai_req.model]
    
    bedrock_args = to_bedrock(
        openai_request=openai_req,
        bedrock_model_id=bedrock_model_id,
    )

    start_time = time.monotonic()

    if openai_req.stream:
        raw_stream = bedrock_stream(bedrock_args)
        translator = SSETranslator(openai_req.model)
        return _stream_with_telemetry(
            translator,
            raw_stream,
            user_context,
            model=openai_req.model,
            bedrock_model_id=bedrock_model_id,
            start_time=start_time,
        )

    bedrock_resp = bedrock_invoke(bedrock_args)
    duration_ms = (time.monotonic() - start_time) * 1000
    logger.info("usage", extra={
        "user": user_context.user_id,
        "model": openai_req.model,
        "providerModel": bedrock_model_id,
        "inputTokens": bedrock_resp.usage.inputTokens,
        "outputTokens": bedrock_resp.usage.outputTokens,
        "durationMs": round(duration_ms, 2),
    })
    return to_openai(bedrock_resp, model=openai_req.model)


def _stream_with_telemetry(translator: SSETranslator, raw_stream, user_context: UserContext,
                           model: str, bedrock_model_id: str, start_time: float):
    # Logs usage after the stream is fully consumed (or on error mid-stream).
    def _gen():
        try:
            yield from translator.translate(raw_stream)
        finally:
            duration_ms = (time.monotonic() - start_time) * 1000
            logger.info("usage", extra={
                "user": user_context.user_id,
                "model": model,
                "providerModel": bedrock_model_id,
                "inputTokens": translator.input_tokens,
                "outputTokens": translator.output_tokens,
                "durationMs": round(duration_ms, 2),
            })

    return _gen()


def _validate_content(req: OpenAIRequest) -> None:
    for msg in req.messages:
        if isinstance(msg.content, list):
            for part in msg.content:
                if part.type != "text":
                    raise ValidationError("Image and document content is not supported")
