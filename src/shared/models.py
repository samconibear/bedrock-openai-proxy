from __future__ import annotations
from pydantic import BaseModel


class UserContext(BaseModel):
    user_id: str
    team: str
    roles: list[str]
