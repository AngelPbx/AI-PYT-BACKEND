from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


# ============== SIP Trunk ==============

class SIPTrunkCreate(BaseModel):
    name: str
    numbers: List[str]
    auth_username: str
    auth_password: str


class SIPTrunkOut(SIPTrunkCreate):
    id: int
    livekit_trunk_id: Optional[str]
    created_at: datetime

    class Config:
        orm_mode = True


# ============== Dispatch Rule ==============

class SIPDispatchRuleCreate(BaseModel):
    name: str
    room_prefix: str
    agent_name: str
    metadata: Optional[str] = None


class SIPDispatchRuleOut(SIPDispatchRuleCreate):
    id: int
    livekit_dispatch_id: Optional[str]
    created_at: datetime

    class Config:
        orm_mode = True


# ============== Call Session ==============

class CallSessionOut(BaseModel):
    id: int
    room_id: str
    caller_number: str
    agent_identity: Optional[str]
    status: str
    created_at: datetime

    class Config:
        orm_mode = True
