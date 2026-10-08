"""Gateway Lambda entry point: route requests, build responses, map exceptions to HTTP errors."""

import json
import traceback

from aws_lambda_powertools import Logger

from src.shared.exceptions import (
    AuthError,
    GatewayError,
    ProviderError,
    RateLimitError,
    RoutingError,
    ValidationError,
)
from src.shared.models import UserContext
from src.gateway.chat.handler import handle as chat_handle
from src.gateway.list_models.handler import handle as models_handle

logger = Logger()


def handler(event, context):
    user_context = _extract_user_context(event)
    method = event.get("requestContext", {}).get("http", {}).get("method", "").upper()
    path = event.get("requestContext", {}).get("http", {}).get("path", "")

    try:
        if method == "GET" and path == "/v1/models":
            return _json_response(200, models_handle())

        if method == "POST" and path == "/v1/chat/completions":
            result = chat_handle(event, user_context)
            if hasattr(result, "__iter__") and not isinstance(result, dict):
                return _streaming_response(result)
            return _json_response(200, result)

        raise RoutingError(f"No route for {method} {path}")

    except AuthError as e:
        return _error_response(e.status_code, e.message)
    except RoutingError as e:
        return _error_response(e.status_code, e.message)
    except ValidationError as e:
        return _error_response(e.status_code, e.message)
    except RateLimitError as e:
        return _error_response(e.status_code, e.message)
    except ProviderError as e:
        logger.error("provider_error", extra={"error": str(e)})
        return _error_response(e.status_code, e.message)
    except GatewayError as e:
        logger.error("gateway_error", extra={"error": str(e)})
        return _error_response(e.status_code, e.message)
    except Exception as e:
        logger.error("unhandled_error", extra={"error": str(e), "trace": traceback.format_exc()})
        return _error_response(500, "Internal server error")


def _extract_user_context(event: dict) -> UserContext:
    authorizer = event.get("requestContext", {}).get("authorizer", {}).get("lambda", {})
    user_id = authorizer.get("userId", "")
    team = authorizer.get("team", "")
    roles_str = authorizer.get("roles", "")
    roles = [r for r in roles_str.split(",") if r] if roles_str else []
    return UserContext(user_id=user_id, team=team, roles=roles)


def _json_response(status_code: int, body: dict) -> dict:
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def _error_response(status_code: int, message: str) -> dict:
    return _json_response(status_code, {"error": {"message": message, "type": "gateway_error"}})


def _streaming_response(generator) -> dict:
    # Requires InvokeMode=RESPONSE_STREAM on the function URL or API Gateway integration.
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
        "body": generator,
    }
