from pydantic import BaseModel, Field, EmailStr
from typing import Optional
from datetime import datetime

class UserSignup(BaseModel):
    username: str = Field(..., min_length=5)
    email: EmailStr
    full_name: str
    password: str
    retall_api_key: Optional[str] = None

class UserLogin(BaseModel):
    username: str
    password: str

class UpdateUser(BaseModel):
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    password: Optional[str] = None

class WorkspaceCreate(BaseModel):
    name: str
    description: Optional[str] = None

class WorkspaceOut(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    created_at: str
    owner_username: str

    class Config:
        from_attributes = True

class InviteMember(BaseModel):
    username: str
    role: Optional[str] = "member"

class KnowledgeBaseCreate(BaseModel):
    name: str
    workspace_id: int

class KnowledgeBaseOut(BaseModel):
    id: str
    name: str
    file_path: str
    created_at: datetime
    workspace_id: int

    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class KnowledgeFileOut(BaseModel):
    id: int
    filename: str
    file_path: str
    kb_id: str
    uploaded_at: datetime
    extract_data: Optional[str] = None

    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class APIKeyCreate(BaseModel):
    name: str

class WorkspaceSettingsUpdate(BaseModel):
    default_voice: Optional[str] = None
    default_model: Optional[str] = None
    temperature: Optional[float] = None
