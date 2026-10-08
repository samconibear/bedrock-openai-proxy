"""
Lambda authorizer for HTTP API Gateway v2 (simple response format).

API Gateway context values must be strings — roles are comma-joined.
300s TTL cache means a revoked key stays valid for up to 5 minutes.
"""

from src.authorizer.validator import validate_key


def handler(event, context):
    token = _extract_token(event)
    if not token:
        return {"isAuthorized": False}

    user_context = validate_key(token)
    if not user_context:
        return {"isAuthorized": False}

    return {
        "isAuthorized": True,
        "context": {
            "userId": user_context.user_id,
            "team": user_context.team,
            "roles": ",".join(user_context.roles),  # API GW context is string-only
        },
    }


def _extract_token(event: dict) -> str | None:
    auth_header = (
        event.get("headers", {}).get("authorization")
        or event.get("headers", {}).get("Authorization")
        or ""
    )
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip() or None
    return None
