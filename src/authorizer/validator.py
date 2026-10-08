import hashlib
import os

from src.authorizer.dynamo import get_client
from src.shared.models import UserContext

TABLE_NAME = os.environ.get("GATEWAY_TABLE_NAME", "gateway")


def validate_key(raw_token: str) -> UserContext | None:
    key_hash = hashlib.sha256(raw_token.encode()).hexdigest()
    pk = f"APIKEY#{key_hash}"

    client = get_client()
    resp = client.get_item(
        TableName=TABLE_NAME,
        Key={
            "PK": {"S": pk},
            "SK": {"S": pk},
        },
    )

    item = resp.get("Item")
    if not item:
        return None

    if item.get("status", {}).get("S") != "active":
        return None

    user_id = item.get("userId", {}).get("S", "")

    profile_resp = client.get_item(
        TableName=TABLE_NAME,
        Key={
            "PK": {"S": f"USER#{user_id}"},
            "SK": {"S": "PROFILE"},
        },
    )
    profile = profile_resp.get("Item", {})
    team = profile.get("team", {}).get("S", "unknown")
    roles_raw = profile.get("roles", {}).get("SS", [])

    return UserContext(user_id=user_id, team=team, roles=list(roles_raw))
