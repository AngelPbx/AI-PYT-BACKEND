import re
import uuid
import pytz
from typing import Optional, Any, List, Dict
from datetime import datetime
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from models.models import WorkspaceMember

def validate_email(email: str):
    if not re.match(r"^[^@]+@[^@]+\.(com)$", email):
        raise HTTPException(status_code=400, detail="Invalid email format")

def validate_password(password: str):
    if not re.match(r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[\W_]).{8,}$", password):
        raise HTTPException(status_code=400, detail="Password too weak")

def format_response(
    status: bool,
    message: str,
    data: Optional[Any] = None,
    errors: Optional[List[Dict[str, str]]] = None,
    status_code: int = 201
) -> dict:
    response = {
        "status": status,
        "message": message,
        "data": data
    }
    if errors is not None:
        response["errors"] = errors
    return response

def format_datetime_ist(dt: datetime) -> str:
    ist = pytz.timezone("Asia/Kolkata")
    if dt.tzinfo is None:
        dt = pytz.utc.localize(dt)
    dt_ist = dt.astimezone(ist)
    return dt_ist.strftime("%Y-%m-%d %H:%M:%S")

def generate_api_key() -> str:
    return f"key_{uuid.uuid4().hex}"

def is_user_in_workspace(user_id: int, workspace_id: int, db: Session) -> bool:
    return db.query(WorkspaceMember).filter_by(user_id=user_id, workspace_id=workspace_id).first() is not None
