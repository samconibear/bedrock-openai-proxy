import time

from src.gateway.shared.model_map import MODEL_MAP


def handle() -> dict:
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": alias,
                "object": "model",
                "created": now,
                "owned_by": "bedrock",
            }
            for alias in MODEL_MAP
        ],
    }
