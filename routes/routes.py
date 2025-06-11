import os, shutil, fitz
from docx import Document  #pip install pymupdf python-docx
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from io import BytesIO
from openai import OpenAI
from fastapi import FastAPI, Depends, Query, HTTPException, Form, UploadFile, File, Body, APIRouter, BackgroundTasks
from utils.helpers import is_user_in_workspace, generate_api_key
from sqlalchemy.orm import Session
import uuid
import requests
from bs4 import BeautifulSoup
import time
from sqlalchemy.orm import sessionmaker, Session
from db.database import get_db, get_current_user
from models.models import User, Workspace, WorkspaceSettings, WorkspaceMember, KnowledgeBase, KnowledgeFile, APIKey, FileStatus, SourceStatus
from models.schemas import (
    UserSignup, UserLogin, UpdateUser,
    WorkspaceCreate, WorkspaceOut, InviteMember,
    WorkspaceSettingsUpdate, KnowledgeBaseCreate
)
from utils.helpers import (
    format_response, validate_email,
    validate_password, format_datetime_ist
)
from utils.security import (
    hash_password, verify_password,
    create_token
)

app = FastAPI()
router = APIRouter()
from db.database import engine, Base
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# UPLOAD_DIR = Path("uploads").resolve()
# UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads")).resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
@router.get("/check-username")

def check_username_availability(
    username: str = Query(..., min_length=5, max_length=50),
    db: Session = Depends(get_db)
):
    try:
        existing_user = db.query(User).filter(User.username == username).first()
        if existing_user:
            return format_response(
                status=False,
                message="Username already taken",
                errors=[{"field": "username", "message": "This username is already in use"}]
            )
        return format_response(
            status=True,
            message="Username is available"
        )

    except Exception as e:
        return format_response(
            status=False,
            message="Internal Server Error",
            errors=[{"field": "server", "message": str(e)}]
        )


@router.post("/signup")
def signup(user: UserSignup, db: Session = Depends(get_db)):
    try:
        # Validate inputs
        validate_email(user.email)
        validate_password(user.password)

        # Check if user already exists
        if db.query(User).filter(User.username == user.username).first():
            return format_response(
                status=False,
                message="Username already exists",
                errors=[{"field": "username", "message": "Username already exists"}]
            )

        if db.query(User).filter(User.email == user.email).first():
            return format_response(
                status=False,
                message="Email already exists",
                errors=[{"field": "email", "message": "Email already exists"}]
            )

        # Create user
        new_user = User(
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            retall_api_key=user.retall_api_key or os.getenv("RETAIL_API_KEY"),
            hashed_password=hash_password(user.password),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)

        # Create default workspace
        workspace = Workspace(
            name=f"{user.username}_workspace",
            description="Default workspace",
            owner_id=new_user.id,
            created_at=datetime.utcnow()
        )
        db.add(workspace)
        db.commit()
        db.refresh(workspace)

        # Add user as member and create workspace settings
        settings = WorkspaceSettings(
            workspace_id=workspace.id,
            default_model="gpt-4",
            default_voice="echo",
            temperature=1
        )
        member = WorkspaceMember(
            user_id=new_user.id,
            workspace_id=workspace.id,
            role="owner"
        )
        db.add_all([settings, member])
        db.commit()

        return format_response(
            status=True,
            message="Signup successful",
            data={
                "retall_api_key": new_user.retall_api_key,
                "workspace_id": workspace.id,
                "workspace_name": workspace.name
            }
        )

    except Exception as e:
        db.rollback()
        return format_response(
            status=False,
            message="Internal Server Error",
            errors=[{"field": "server", "message": str(e)}]
        )


@router.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    try:
        db_user = db.query(User).filter(User.username == user.username).first()
        if not db_user or not verify_password(user.password, db_user.hashed_password):
            raise HTTPException(status_code=401, detail="Invalid credentials")

        token, expire = create_token({"username": db_user.username, "expire_minutes": os.getenv("TOKEN_EXPIRE_MINUTES", 60)})

        return format_response(
            status=True,
            message="Login successful",
            data={
                "token": 'Bearer_' + token,
                "expires_at": expire.isoformat() + "Z",
                "expire_duration": int(os.getenv("TOKEN_EXPIRE_MINUTES")),
                "retall_api_key": db_user.retall_api_key
            }
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@router.patch("/user/update")
def update_user(
    user_data: UpdateUser,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        existing_user = db.query(User).filter(User.id == current_user.id).first()
        if not existing_user:
            return format_response(
                status=False,
                message="User not found",
                errors=[{"field": "user", "message": "User not found"}]
            )

        updated = False
        errors = []

        # Update username
        if user_data.username and user_data.username != existing_user.username:
            username_exists = db.query(User).filter(User.username == user_data.username).first()
            if username_exists:
                errors.append({
                    "field": "username",
                    "message": "Username already taken"
                })
            else:
                existing_user.username = user_data.username
                updated = True

        # Update email
        if user_data.email and user_data.email != existing_user.email:
            try:
                validate_email(user_data.email)
            except Exception as e:
                errors.append({
                    "field": "email",
                    "message": str(e)
                })
            else:
                email_exists = db.query(User).filter(User.email == user_data.email).first()
                if email_exists:
                    errors.append({
                        "field": "email",
                        "message": "Email already registered"
                    })
                else:
                    existing_user.email = user_data.email
                    updated = True

        # Update password
        if user_data.password:
            try:
                validate_password(user_data.password)
            except Exception as e:
                errors.append({
                    "field": "password",
                    "message": str(e)
                })
            else:
                existing_user.hashed_password = hash_password(user_data.password)
                updated = True

        if errors:
            return format_response(
                status=False,
                message="Validation or conflict error",
                errors=errors
            )

        if not updated:
            return format_response(
                status=False,
                message="No valid fields provided for update",
                errors=[{"field": "update", "message": "Nothing to update"}]
            )

        # Update the timestamp
        existing_user.updated_at = datetime.utcnow()

        db.commit()
        user_response = {
            "id": existing_user.id,
            "username": existing_user.username,
            "email": existing_user.email,
            "full_name": existing_user.full_name,
            # Add other fields you want to return here
        }

        return format_response(
            status=True,
            message="User details updated successfully",
            data=user_response
        )

    except Exception as e:
        db.rollback()
        return format_response(
            status=False,
            message="Internal Server Error",
            errors=[{"field": "server", "message": str(e)}]
        )
# @router.put("/user/update")
# def update_user(
#     user_data: UpdateUser,
#     db: Session = Depends(get_db),
#     current_user: User = Depends(get_current_user)
# ):
#     try:
#         existing_user = db.query(User).filter(User.id == current_user.id).first()
#         if not existing_user:
#             return format_response(
#                 status=False,
#                 message="User not found",
#                 errors=[{"field": "user", "message": "User not found"}]
#             )

#         updated = False
#         errors = []

#         # Update username
#         if user_data.username and user_data.username != existing_user.username:
#             username_exists = db.query(User).filter(User.username == user_data.username).first()
#             if username_exists:
#                 errors.append({
#                     "field": "username",
#                     "message": "Username already taken"
#                 })
#             else:
#                 existing_user.username = user_data.username
#                 updated = True

#         # Update email
#         if user_data.email and user_data.email != existing_user.email:
#             try:
#                 validate_email(user_data.email)
#             except Exception as e:
#                 errors.append({
#                     "field": "email",
#                     "message": str(e)
#                 })
#             else:
#                 email_exists = db.query(User).filter(User.email == user_data.email).first()
#                 if email_exists:
#                     errors.append({
#                         "field": "email",
#                         "message": "Email already registered"
#                     })
#                 else:
#                     existing_user.email = user_data.email
#                     updated = True

#         # Update password
#         if user_data.password:
#             try:
#                 validate_password(user_data.password)
#             except Exception as e:
#                 errors.append({
#                     "field": "password",
#                     "message": str(e)
#                 })
#             else:
#                 existing_user.hashed_password = hash_password(user_data.password)
#                 updated = True

#         if errors:
#             return format_response(
#                 status=False,
#                 message="Validation or conflict error",
#                 errors=errors
#             )

#         if not updated:
#             return format_response(
#                 status=False,
#                 message="No valid fields provided for update",
#                 errors=[{"field": "update", "message": "Nothing to update"}]
#             )

#         db.commit()

#         return format_response(
#             status=True,
#             message="User details updated successfully",
#             data={}
#         )

#     except Exception as e:
#         db.rollback()
#         return format_response(
#             status=False,
#             message="Internal Server Error",
#             errors=[{"field": "server", "message": str(e)}]
#         )


@router.get("/user/status")
def check_user_status(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        user = db.query(User).filter(User.id == current_user.id).first()

        if not user:
            return format_response(
                status=False,
                message="User not found",
                errors=[{"field": "user", "message": "User not found"}]
            )

        return format_response(
            status=True,
            message="User is active" if user.is_active else "User is inactive",
            data={"is_active": user.is_active}
        )

    except Exception as e:
        return format_response(
            status=False,
            message="Internal Server Error",
            errors=[{"field": "server", "message": str(e)}]
        )


@router.post("/workspaces")
def create_workspace(data: WorkspaceCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        workspace = Workspace(
            name=data.name,
            description=data.description,
            owner_id=current_user.id
        )
        db.add(workspace)
        db.commit()
        db.refresh(workspace)

        settings = WorkspaceSettings(
            workspace_id=workspace.id,
            default_model="gpt-4",
            default_voice="echo",
            temperature=1
        )
        db.add(settings)

        member = WorkspaceMember(
            user_id=current_user.id,
            workspace_id=workspace.id,
            role="owner"
        )
        db.add(member)
        db.commit()

        return format_response(
            status=True,
            message="Workspace created",
            data=WorkspaceOut(
                id=workspace.id,
                name=workspace.name,
                description=workspace.description,
                created_at=format_datetime_ist(workspace.created_at),
                owner_username=current_user.username,
            ).dict()
        )

    except Exception as e:
        db.rollback()
        return format_response(
            status=False,
            message="Internal Server Error",
            errors=[{"field": "server", "message": str(e)}]
        )


@router.get("/workspaces")
def list_workspaces(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        workspaces = (
            db.query(Workspace)
            .join(WorkspaceMember)
            .filter(WorkspaceMember.user_id == current_user.id)
            .all()
        )

        data = [
            WorkspaceOut(
                id=w.id,
                name=w.name,
                created_at=format_datetime_ist(w.created_at),
                owner_username=w.owner.username
            ).dict() for w in workspaces
        ]

        return format_response(
            status=True,
            message="Workspaces retrieved successfully",
            data=data
        )

    except Exception as e:
        return format_response(
            status=False,
            message="Internal Server Error",
            errors=[{"field": "server", "message": str(e)}]
        )

@router.post("/workspaces/{workspace_id}/invite")
def invite_user(
    workspace_id: int,
    invite: InviteMember,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        workspace = db.query(Workspace).filter(Workspace.id == workspace_id).first()
        if not workspace:
            return format_response(
                status=False,
                message="Workspace not found",
                errors=[{"field": "workspace_id", "message": "Workspace not found"}]
            )

        if workspace.owner_id != current_user.id:
            return format_response(
                status=False,
                message="Only the owner can invite members",
                errors=[{"field": "permission", "message": "Only the owner can invite"}],
                status_code=403
            )

        user = db.query(User).filter(User.username == invite.username).first()
        if not user:
            return format_response(
                status=False,
                message="User not found",
                errors=[{"field": "username", "message": "User not found"}]
            )

        existing = db.query(WorkspaceMember).filter_by(workspace_id=workspace_id, user_id=user.id).first()
        if existing:
            return format_response(
                status=False,
                message="User already a member",
                errors=[{"field": "username", "message": "Already a member"}],
                status_code=400
            )

        new_member = WorkspaceMember(workspace_id=workspace_id, user_id=user.id, role=invite.role)
        db.add(new_member)
        db.commit()

        return format_response(
            status=True,
            message=f"{invite.username} invited to workspace"
        )

    except Exception as e:
        db.rollback()
        return format_response(
            status=False,
            message="Internal Server Error",
            errors=[{"field": "server", "message": str(e)}]
        )

@router.delete("/workspaces/{workspace_id}")
def delete_workspace(
    workspace_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        workspace = db.query(Workspace).filter(Workspace.id == workspace_id).first()
        if not workspace:
            return format_response(
                status=False,
                message="Workspace not found",
                errors=[{"field": "workspace_id", "message": "Workspace not found"}],
                status_code=404
            )

        if workspace.owner_id != current_user.id:
            return format_response(
                status=False,
                message="Only the owner can delete the workspace",
                errors=[{"field": "permission", "message": "Unauthorized"}],
                status_code=403
            )

        # Delete knowledge base folders from filesystem
        for kb in workspace.knowledge_bases:
            kb_path = UPLOAD_DIR / str(workspace.id) / str(kb.id)
            if kb_path.exists():
                import shutil
                shutil.rmtree(kb_path)

        # Delete workspace (with cascading)
        db.delete(workspace)
        db.commit()

        return format_response(
            status=True,
            message="Workspace deleted successfully"
        )

    except Exception as e:
        db.rollback()
        return format_response(
            status=False,
            message="Internal Server Error",
            errors=[{"field": "server", "message": str(e)}]
        )

@router.post("/workspaces/{workspace_id}/knowledge-bases")
def create_knowledge_base(
    workspace_id: int,
    source_type: str = Form(...),  # 'file', 'web_page', or 'text'
    name: str = Form(...),
    file: UploadFile = File(None),
    url: str = Form(None),
    text_filename: str = Form(None),
    text_content: str = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        workspace = db.query(Workspace).filter(
            Workspace.id == workspace_id,
            Workspace.owner_id == current_user.id
        ).first()

        if not workspace:
            return format_response(
                status=False,
                message="Workspace not found",
                errors=[{"field": "workspace_id", "message": "Workspace not found"}],
                status_code=404
            )
            
        if not name.strip():
            return format_response(
                status=False,
                message="Validation error",
                errors=[{"field": "name", "message": "Knowledge base name is required"}],
                status_code=400
            )

        kb_id_str = f"knowledge_base_{uuid.uuid4().hex[:16]}"
        kb_folder = UPLOAD_DIR / f"workspace_{workspace_id}" / kb_id_str
        kb_folder.mkdir(parents=True, exist_ok=True)

        now_utc = datetime.utcnow()
        file_path = None
        file_name = None
        source_details = {}

        if source_type == "file":
            if not file:
                return format_response(
                    status=False,
                    message="File upload required",
                    errors=[{"field": "file", "message": "No file uploaded"}]
                )
            file_name = f"{uuid.uuid4().hex}_{file.filename}"
            file_path = kb_folder / file_name
            with open(file_path, "wb") as buffer:
                buffer.write(file.file.read())
            source_details = {
                "type": "document",
                "source_id": str(workspace_id),
                "filename": file_name,
                "file_url": str(file_path)
            }

        elif source_type == "web_page":
            if not url:
                return format_response(
                    status=False,
                    message="URL required",
                    errors=[{"field": "url", "message": "No URL provided"}]
                )
            file_name = f"web_{uuid.uuid4().hex}.txt"
            file_path = kb_folder / file_name
           
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"Crawled content from {url}")  # to complet crawler logic
            source_details = {
                "type": "web_page",
                "source_id": str(workspace_id),
                "filename": file_name,
                "file_url": str(file_path)
            }

        elif source_type == "text":
            if not text_filename or not text_content:
                return format_response(
                    status=False,
                    message="Text filename and content required",
                    errors=[{"field": "text", "message": "Missing text filename or content"}]
                )
            file_name = f"{uuid.uuid4().hex}_{text_filename}.txt"
            file_path = kb_folder / file_name
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text_content)
            source_details = {
                "type": "text",
                "source_id": str(workspace_id),
                "filename": file_name,
                "file_url": str(file_path)
            }

        else:
            return format_response(
                status=False,
                message="Invalid source type",
                errors=[{"field": "source_type", "message": "Must be one of: file, web_page, text"}]
            )

        kb = KnowledgeBase(
            name=name,
            file_path=str(file_path),
            workspace_id=workspace_id,
            enable_auto_refresh=True,
            auto_refresh_interval=24,
            last_refreshed=now_utc
        )
        db.add(kb)
        db.commit()
        db.refresh(kb)

        response_data = {
            "knowledge_base_id": kb_id_str,
            "knowledge_base_name": kb.name,
            "status": "in_progress",
            "knowledge_base_sources": [source_details],
            "enable_auto_refresh": kb.enable_auto_refresh,
            "last_refreshed_timestamp": int(now_utc.timestamp() * 1000)
        }

        return format_response(
            status=True,
            message="Knowledge base created successfully",
            data=response_data
        )

    except Exception as e:
        db.rollback()
        return format_response(
            status=False,
            message="Internal Server Error",
            errors=[{"field": "server", "message": str(e)}]
        )

@router.get("/workspaces/{workspace_id}/knowledge-bases")
def list_kbs(
    workspace_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        membership = db.query(WorkspaceMember).filter_by(
            workspace_id=workspace_id,
            user_id=current_user.id
        ).first()

        if not membership:
            return format_response(
                status=False,
                message="Access denied",
                errors=[{"field": "workspace_id", "message": "User is not a member of this workspace"}],
                status_code=403
            )

        kbs = db.query(KnowledgeBase).filter_by(workspace_id=workspace_id).all()

        data = [
            {
                "id": kb.id,
                "name": kb.name,
                "file_path": kb.file_path,
                "workspace_id": kb.workspace_id,
                "created_at": format_datetime_ist(kb.created_at)
            } for kb in kbs
        ]

        return format_response(
            status=True,
            message="Knowledge bases retrieved successfully",
            data=data
        )

    except Exception as e:
        db.rollback()
        return format_response(
            status=False,
            message="Internal Server Error",
            errors=[{"field": "server", "message": str(e)}]
        )

# To create a knowledge base, we need to ensure the user is a member of the workspace
@router.post("/knowledge-bases")
def create_knowledge_base(
    kb_data: KnowledgeBaseCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        # Validate workspace membership or ownership here
        membership = db.query(WorkspaceMember).filter_by(
            workspace_id=kb_data.workspace_id,
            user_id=current_user.id
        ).first()
        if not membership:
            return format_response(
                status=False,
                message="Access denied",
                errors=[{"field": "workspace_id", "message": "User is not a member of this workspace"}],
                status_code=403
            )

        # Generate a unique string ID for the knowledge base
        # kb_id = uuid4().hex
        # print(f"Generated KB ID: {kb_id}")
        print(f"Knowledge Base Data: {kb_data}")
        kb = KnowledgeBase(
            # id=kb_id,
            name=kb_data.name,
            file_path="",
            workspace_id=kb_data.workspace_id,
            created_at=datetime.utcnow(),
            enable_auto_refresh=False,  # default or configurable
            last_refreshed=None
        )
        db.add(kb)
        db.commit()
        db.refresh(kb)

        return format_response(
            status=True,
            message="Knowledge base created successfully",
            data={
                "id": kb.id,
                "name": kb.name,
                "workspace_id": kb.workspace_id,
                "created_at": kb.created_at.isoformat()
            }
        )
    except Exception as e:
        db.rollback()
        return format_response(
            status=False,
            message="Internal Server Error",
            errors=[{"field": "server", "message": str(e)}],
            status_code=500
        )
    
@router.get("/knowledge-bases/{kb_id}")
def get_knowledge_base(
    kb_id: str,  # changed from int to str
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
        if not kb:
            return format_response(
                status=False,
                message="Knowledge base not found",
                errors=[{"field": "kb_id", "message": "Knowledge base not found"}],
                status_code=404
            )

        is_member = db.query(WorkspaceMember).filter_by(
            workspace_id=kb.workspace_id,
            user_id=current_user.id
        ).first()

        if not is_member:
            return format_response(
                status=False,
                message="Access denied",
                errors=[{"field": "workspace", "message": "Access denied"}],
                status_code=403
            )

        files = db.query(KnowledgeFile).filter_by(kb_id=kb.id).all()
        sources = [
            {
                "type": "document",
                "source_id": kb.workspace_id,
                "filename": f.filename,
                "file_url": f.file_path
            } for f in files
        ]

        data = {
            "knowledge_base_id": kb.id,
            "knowledge_base_name": kb.name,
            "status": "ready",
            "knowledge_base_sources": sources,
            "enable_auto_refresh": kb.enable_auto_refresh,
            "last_refreshed_timestamp": int(kb.last_refreshed.timestamp() * 1000) if kb.last_refreshed else None
        }

        return format_response(
            status=True,
            message="Knowledge base details fetched",
            data=data
        )

    except Exception as e:
        db.rollback()
        return format_response(
            status=False,
            message="Internal Server Error",
            errors=[{"field": "server", "message": str(e)}],
            status_code=500
        )
       
from openai import OpenAI

# Ensure you initialize OpenAI client
openai_client = OpenAI()

@router.post("/knowledge-bases/sources")
async def add_kb_source(
    kb_id: str = Form(...),
    source_type: str = Form(...),
    file: UploadFile = File(None),
    url: str = Form(None),
    text_filename: str = Form(None),
    text_content: str = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        # Validate KB and user membership
        kb = db.query(KnowledgeBase).filter_by(id=kb_id).first()
        if not kb:
            return format_response(False, "Knowledge base not found", errors=[{"field": "kb_id"}], status_code=404)

        member = db.query(WorkspaceMember).filter_by(workspace_id=kb.workspace_id, user_id=current_user.id).first()
        if not member:
            return format_response(False, "Access denied", errors=[{"field": "workspace"}], status_code=403)

        # Directory setup
        kb_path = UPLOAD_DIR / f"workspace_{kb.workspace_id}" / f"knowledge_base_{kb_id}"
        kb_path.mkdir(parents=True, exist_ok=True)

        # Process content
        file_path, file_name, extract_text = None, None, ""

        if source_type == "file":
            if not file:
                return format_response(False, "File is required", errors=[{"field": "file"}], status_code=400)

            ext = Path(file.filename).suffix.lower()
            file_name = f"{uuid4().hex}_{file.filename}"
            file_path = kb_path / file_name

            with open(file_path, "wb") as f_out:
                f_out.write(await file.read())

            if ext == ".pdf":
                with fitz.open(file_path) as doc:
                    extract_text = "\n".join(p.get_text() for p in doc)
            elif ext == ".docx":
                extract_text = "\n".join(p.text for p in Document(file_path).paragraphs)
            elif ext == ".txt":
                extract_text = file_path.read_text(encoding="utf-8")
            else:
                return format_response(False, "Unsupported file type", status_code=400)

        elif source_type == "web_page":
            if not url or not text_filename:
                return format_response(False, "URL and filename required", status_code=400)
            res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            res.raise_for_status()
            soup = BeautifulSoup(res.text, "html.parser")
            for tag in soup(["script", "style"]): tag.decompose()
            extract_text = soup.get_text(separator="\n", strip=True)
            file_name = text_filename
            file_path = url

        elif source_type == "txt":
            if not text_filename or not text_content:
                return format_response(False, "Text filename and content required", status_code=400)
            file_name = f"{uuid4().hex}_{text_filename}.txt"
            file_path = kb_path / file_name
            file_path.write_text(text_content, encoding="utf-8")
            extract_text = text_content

        else:
            return format_response(False, "Invalid source type", status_code=400)

        # ✅ Generate Embedding (truncated for safety)
        clean_text = extract_text.replace("\n", " ").strip()
        truncated = clean_text[:3000]  # limit tokens for embedding

        embedding_response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=truncated
        )
        embedding_vector = embedding_response.data[0].embedding

        # Save to DB
        kb_file = KnowledgeFile(
            kb_id=kb.id,
            filename=file_name,
            file_path=str(file_path),
            extract_data=extract_text,
            embedding=embedding_vector,  # ➕ store embedding
            status=FileStatus.completed,
            source_type=SourceStatus(source_type)
        )
        db.add(kb_file)
        db.commit()

        return format_response(True, "Source added with embedding", data={
            "source_id": kb_file.id,
            "filename": file_name,
            "file_url": str(file_path),
            "type": source_type,
        })

    except Exception as e:
        db.rollback()
        return format_response(False, "Internal Server Error", errors=[{"field": "server", "message": str(e)}], status_code=500)

# @router.post("/knowledge-bases/sources")
# async def add_kb_source(
#     kb_id: str,
#     source_type: str = Form(...),
#     file: UploadFile = File(None),
#     url: str = Form(None),
#     text_filename: str = Form(None),
#     text_content: str = Form(None),
#     db: Session = Depends(get_db),
#     current_user: User = Depends(get_current_user)
# ):
#     try:
#         # Validate KB existence
#         kb = db.query(KnowledgeBase).filter_by(id=kb_id).first()
#         if not kb:
#             return format_response(False, "Knowledge base not found", errors=[{"field": "kb_id"}], status_code=404)

#         # Check workspace membership
#         membership = db.query(WorkspaceMember).filter_by(workspace_id=kb.workspace_id, user_id=current_user.id).first()
#         if not membership:
#             return format_response(False, "Access denied", errors=[{"field": "workspace"}], status_code=403)

#         kb_path = UPLOAD_DIR / f"workspace_{kb.workspace_id}" / f"knowledge_base_{kb_id}"
#         kb_path.mkdir(parents=True, exist_ok=True)

#         source_details = {}

#         if source_type == "file":
#             if not file:
#                 return format_response(False, "File is required", errors=[{"field": "file"}], status_code=400)

#             file_name = f"{uuid4().hex}_{file.filename}"
#             file_path = kb_path / file_name
#             ext = Path(file.filename).suffix.lower()
#             extract_text = ""

#             try:
#                 with open(file_path, "wb") as f_out:
#                     content = await file.read()
#                     f_out.write(content)

#                 if ext == ".pdf":
#                     with fitz.open(file_path) as doc:
#                         extract_text = "\n".join(page.get_text() for page in doc)

#                 elif ext == ".docx":
#                     doc = Document(file_path)
#                     extract_text = "\n".join(p.text for p in doc.paragraphs)

#                 elif ext == ".txt":
#                     with open(file_path, "r", encoding="utf-8") as f:
#                         extract_text = f.read()
#                 else:
#                     raise ValueError(f"Unsupported file type: {ext}")

#                 kb_file = KnowledgeFile(
#                     kb_id=kb.id,
#                     filename=file_name,
#                     file_path=str(file_path),
#                     extract_data=extract_text,
#                     status=FileStatus.completed,
#                     source_type=SourceStatus.file
#                 )
#                 db.add(kb_file)
#                 db.commit()

#                 source_details = {
#                     "type": "document",
#                     "source_id": kb.workspace_id,
#                     "filename": file_name,
#                     "file_url": str(file_path)
#                 }

#             except Exception as e:
#                 db.rollback()
#                 if file_path.exists():
#                     file_path.unlink(missing_ok=True)
#                 return format_response(False, "File processing failed", errors=[{"field": "file", "message": str(e)}], status_code=400)

#         elif source_type == "web_page":
#             if not url or not text_filename:
#                 return format_response(False, "URL and filename required", errors=[{"field": "url"}, {"field": "text_filename"}], status_code=400)

#             try:
#                 headers = {"User-Agent": "Mozilla/5.0"}
#                 response = requests.get(url, headers=headers, timeout=10)
#                 response.raise_for_status()
#                 soup = BeautifulSoup(response.text, "html.parser")

#                 for tag in soup(["script", "style"]):
#                     tag.decompose()

#                 cleaned_text = soup.get_text(separator="\n", strip=True).replace('\n', ' ')

#                 kb_file = KnowledgeFile(
#                     kb_id=kb.id,
#                     filename=text_filename,
#                     file_path=url,
#                     extract_data=cleaned_text,
#                     status=FileStatus.completed,
#                     source_type=SourceStatus.url
#                 )
#                 db.add(kb_file)
#                 db.commit()

#                 source_details = {
#                     "type": "web_page",
#                     "source_id": kb.workspace_id,
#                     "filename": text_filename,
#                     "file_url": url
#                 }

#             except Exception as e:
#                 db.rollback()
#                 return format_response(False, "Failed to fetch or parse URL", errors=[{"field": "url", "message": str(e)}], status_code=400)

#         elif source_type == "text":
#             if not text_filename or not text_content:
#                 return format_response(False, "Text filename and content required", errors=[{"field": "text"}], status_code=400)

#             file_name = f"{uuid4().hex}_{text_filename}.txt"
#             file_path = kb_path / file_name

#             try:
#                 with open(file_path, "w", encoding="utf-8") as f:
#                     f.write(text_content)

#                 kb_file = KnowledgeFile(
#                     kb_id=kb.id,
#                     filename=text_filename,
#                     file_path=str(file_path),
#                     extract_data=text_content,
#                     status=FileStatus.completed,
#                     source_type=SourceStatus.txt
#                 )
#                 db.add(kb_file)
#                 db.commit()

#                 source_details = {
#                     "type": "text",
#                     "source_id": kb.workspace_id,
#                     "filename": text_filename,
#                     "file_url": str(file_path)
#                 }

#             except Exception as e:
#                 db.rollback()
#                 return format_response(False, "Failed to write text to file", errors=[{"field": "text", "message": str(e)}], status_code=500)

#         else:
#             return format_response(False, "Invalid source type", errors=[{"field": "source_type"}], status_code=400)

#         return format_response(True, "Knowledge base source added successfully", data=source_details)

#     except Exception as e:
#         db.rollback()
#         return format_response(False, "Internal Server Error", errors=[{"field": "server", "message": str(e)}], status_code=500)

    
@router.delete("/knowledge-bases/{kb_id}")
def delete_knowledge_base(
    kb_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        # Fetch knowledge base
        kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
        if not kb:
            return format_response(
                status=False,
                message="Knowledge base not found",
                errors=[{"field": "kb_id", "message": "Knowledge base not found"}],
                status_code=404
            )

        # Check workspace ownership
        workspace = db.query(Workspace).filter(Workspace.id == kb.workspace_id).first()
        if not workspace or workspace.owner_id != current_user.id:
            return format_response(
                status=False,
                message="Only the workspace owner can delete this knowledge base",
                errors=[{"field": "permission", "message": "Access denied"}],
                status_code=403
            )

        # Delete related knowledge files
        db.query(KnowledgeFile).filter(KnowledgeFile.kb_id == kb_id).delete()

        # Delete knowledge base
        db.delete(kb)
        db.commit()

        # Remove uploaded files from disk
        kb_path = UPLOAD_DIR / f"workspace_{workspace.id}" / f"knowledge_base_{kb.id}"
        if kb_path.exists():
            shutil.rmtree(kb_path)

        return format_response(
            status=True,
            message="Knowledge base deleted successfully"
        )

    except Exception as e:
        db.rollback()
        return format_response(
            status=False,
            message="Internal Server Error",
            errors=[{"field": "server", "message": str(e)}],
            status_code=500
        )
    
@router.delete("/knowledge-files/{file_id}")
def delete_knowledge_file(
    file_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        # Fetch the file entry
        file = db.query(KnowledgeFile).filter(KnowledgeFile.id == file_id).first()
        if not file:
            return format_response(
                status=False,
                message="File not found",
                errors=[{"field": "file_id", "message": "Knowledge file not found"}],
                status_code=404
            )

        # Check if user has access to the knowledge base
        kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == file.kb_id).first()
        if not kb:
            return format_response(
                status=False,
                message="Knowledge base not found",
                errors=[{"field": "kb_id", "message": "Associated knowledge base not found"}],
                status_code=404
            )

        membership = db.query(WorkspaceMember).filter_by(
            workspace_id=kb.workspace_id,
            user_id=current_user.id
        ).first()

        if not membership:
            return format_response(
                status=False,
                message="Access denied",
                errors=[{"field": "workspace", "message": "User is not a member of this workspace"}],
                status_code=403
            )

        # Attempt to delete the file from the filesystem
        file_path = Path(file.file_path)
        if file_path.exists() and file_path.is_file():
            try:
                file_path.unlink()
            except Exception as e:
                # Log warning or handle non-fatal file deletion error if needed
                pass

        # Delete the file record from the database
        db.delete(file)
        db.commit()

        return format_response(
            status=True,
            message="Knowledge file deleted successfully"
        )

    except Exception as e:
        db.rollback()
        return format_response(
            status=False,
            message="Internal Server Error",
            errors=[{"field": "server", "message": str(e)}],
            status_code=500
        )
# /////////////////////////////////////////////////////////////////////////////////////////////
        
# @router.get("/knowledge-bases/{kb_id}")
# def get_knowledge_base(
#     kb_id: int,
#     db: Session = Depends(get_db),
#     current_user: User = Depends(get_current_user)
# ):
#     try:
#         kb = db.query(KnowledgeBase).filter_by(id=kb_id).first()
#         if not kb:
#             return format_response(
#                 status=False,
#                 message="Knowledge base not found",
#                 errors=[{"field": "kb_id", "message": "Knowledge base not found"}],
#                 status_code=404
#             )

#         membership = db.query(WorkspaceMember).filter_by(
#             workspace_id=kb.workspace_id,
#             user_id=current_user.id
#         ).first()
#         if not membership:
#             return format_response(
#                 status=False,
#                 message="Access denied",
#                 errors=[{"field": "workspace", "message": "Access denied"}],
#                 status_code=403
#             )

#         files = db.query(KnowledgeFile).filter_by(kb_id=kb.id).all()
#         sources = [
#             {
#                 "type": "document",
#                 "source_id": kb.workspace_id,
#                 "filename": f.filename,
#                 "file_url": f.file_path
#             } for f in files
#         ]

#         data = {
#             "knowledge_base_id": f"knowledge_base_{kb.id}",
#             "knowledge_base_name": kb.name,
#             "status": "ready",
#             "knowledge_base_sources": sources,
#             "enable_auto_refresh": kb.enable_auto_refresh,
#             "last_refreshed_timestamp": int(kb.last_refreshed.timestamp() * 1000)
#         }

#         return format_response(
#             status=True,
#             message="Knowledge base details fetched",
#             data=data
#         )

#     except Exception as e:
#         db.rollback()
#         return format_response(
#             status=False,
#             message="Internal Server Error",
#             errors=[{"field": "server", "message": str(e)}]
#         )


# @router.post("/knowledge-bases/{kb_id}/sources")
# async def add_kb_source(
#     kb_id: int,
#     source_type: str = Form(...),
#     file: UploadFile = File(None),
#     url: str = Form(None),
#     text_filename: str = Form(None),
#     text_content: str = Form(None),
#     db: Session = Depends(get_db),
#     current_user: User = Depends(get_current_user)
# ):
#     try:
#         kb = db.query(KnowledgeBase).filter_by(id=kb_id).first()
#         if not kb:
#             return format_response(
#                 status=False,
#                 message="Knowledge base not found",
#                 errors=[{"field": "kb_id", "message": "Knowledge base not found"}],
#                 status_code=404
#             )

#         membership = db.query(WorkspaceMember).filter_by(workspace_id=kb.workspace_id, user_id=current_user.id).first()
#         if not membership:
#             return format_response(
#                 status=False,
#                 message="Access denied",
#                 errors=[{"field": "workspace_id", "message": "User is not a member of this workspace"}],
#                 status_code=403
#             )

#         kb_path = UPLOAD_DIR / f"workspace_{kb.workspace_id}" / f"knowledge_base_{kb_id}"
#         kb_path.mkdir(parents=True, exist_ok=True)

#         file_path = None
#         file_name = None
#         source_details = {}

#         if source_type == "file":
#             if not file:
#                 return format_response(
#                     status=False,
#                     message="No file uploaded",
#                     errors=[{"field": "file", "message": "File required"}]
#                 )
#             file_name = f"{uuid.uuid4().hex}_{file.filename}"
#             file_path = kb_path / file_name
#             with open(file_path, "wb") as f_out:
#                 f_out.write(await file.read())
#             source_details = {
#                 "type": "document",
#                 "source_id": kb.workspace_id,
#                 "filename": file_name,
#                 "file_url": str(file_path)
#             }

#         elif source_type == "web_page":
#             if not url:
#                 return format_response(
#                     status=False,
#                     message="No URL provided",
#                     errors=[{"field": "url", "message": "URL is required"}]
#                 )
#             file_name = f"web_{uuid.uuid4().hex}.txt"
#             file_path = kb_path / file_name
#             with open(file_path, "w", encoding="utf-8") as f:
#                 f.write(f"Crawled content from {url}")
#             source_details = {
#                 "type": "web_page",
#                 "source_id": kb.workspace_id,
#                 "filename": file_name,
#                 "file_url": str(file_path)
#             }

#         elif source_type == "text":
#             if not text_filename or not text_content:
#                 return format_response(
#                     status=False,
#                     message="Filename and content required",
#                     errors=[{"field": "text", "message": "Both filename and content are required"}]
#                 )
#             file_name = f"{uuid.uuid4().hex}_{text_filename}.txt"
#             file_path = kb_path / file_name
#             with open(file_path, "w", encoding="utf-8") as f:
#                 f.write(text_content)
#             source_details = {
#                 "type": "text",
#                 "source_id": kb.workspace_id,
#                 "filename": file_name,
#                 "file_url": str(file_path)
#             }

#         else:
#             return format_response(
#                 status=False,
#                 message="Invalid source type",
#                 errors=[{"field": "source_type", "message": "Must be one of: file, web_page, text"}],
#                 status_code=400
#             )

#         # Optionally store source in KnowledgeFile
#         if source_type == "file":
#             kb_file = KnowledgeFile(
#                 kb_id=kb.id,
#                 filename=file_name,
#                 file_path=str(file_path)
#             )
#             db.add(kb_file)
#             db.commit()
#             db.refresh(kb_file)

#         return format_response(
#             status=True,
#             message="Knowledge base source added",
#             data=source_details
#         )

#     except Exception as e:
#         db.rollback()
#         return format_response(
#             status=False,
#             message="Internal Server Error",
#             errors=[{"field": "server", "message": str(e)}]
#         )

# @router.delete("/knowledge-bases/{kb_id}")
# def delete_kb(
#     kb_id: int,
#     db: Session = Depends(get_db),
#     current_user: User = Depends(get_current_user)
# ):
#     try:
#         kb = db.query(KnowledgeBase).filter_by(id=kb_id).first()
#         if not kb:
#             return format_response(
#                 status=False,
#                 message="Knowledge base not found",
#                 errors=[{"field": "kb_id", "message": "Knowledge base not found"}],
#                 status_code=404
#             )

#         workspace = db.query(Workspace).filter_by(id=kb.workspace_id).first()
#         if workspace.owner_id != current_user.id:
#             print(f"DEBUG: current_user.id = {current_user.id}, workspace.owner_id = {workspace.owner_id}")
#             return format_response(
#                 status=False,
#                 message="Only owner can delete",
#                 errors=[{"field": "permission", "message": "Only the workspace owner can delete this knowledge base"}],
#                 status_code=403
#             )

#         db.query(KnowledgeFile).filter_by(kb_id=kb_id).delete()
#         db.delete(kb)
#         db.commit()

#         kb_path = UPLOAD_DIR / str(workspace.id) / str(kb.id)
#         if kb_path.exists():
#             import shutil
#             shutil.rmtree(kb_path)

#         return format_response(
#             status=True,
#             message="Knowledge base deleted"
#         )

#     except Exception as e:
#         db.rollback()
#         return format_response(
#             status=False,
#             message="Internal Server Error",
#             errors=[{"field": "server", "message": str(e)}]
#         )

# @router.delete("/knowledge-files/{file_id}")
# def delete_kb_file(
#     file_id: int,
#     db: Session = Depends(get_db),
#     current_user: User = Depends(get_current_user)
# ):
#     try:
#         file = db.query(KnowledgeFile).filter_by(id=file_id).first()
#         if not file:
#             return format_response(
#                 status=False,
#                 message="File not found",
#                 errors=[{"field": "file_id", "message": "Knowledge file not found"}],
#                 status_code=404
#             )

#         kb = db.query(KnowledgeBase).filter_by(id=file.kb_id).first()
#         membership = db.query(WorkspaceMember).filter_by(workspace_id=kb.workspace_id, user_id=current_user.id).first()
#         if not membership:
#             return format_response(
#                 status=False,
#                 message="Access denied",
#                 errors=[{"field": "workspace_id", "message": "User is not a member of this workspace"}],
#                 status_code=403
#             )

#         try:
#             os.remove(file.file_path)
#         except:
#             pass

#         db.delete(file)
#         db.commit()

#         return format_response(
#             status=True,
#             message="File deleted"
#         )

#     except Exception as e:
#         db.rollback()
#         return format_response(
#             status=False,
#             message="Internal Server Error",
#             errors=[{"field": "server", "message": str(e)}]
#         )
    
@router.get("/workspaces/{workspace_id}/settings")
def get_workspace_settings(
    workspace_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        membership = db.query(WorkspaceMember).filter_by(workspace_id=workspace_id, user_id=current_user.id).first()
        if not membership:
            return format_response(
                status=False,
                message="Access denied",
                errors=[{"field": "workspace_id", "message": "User is not a member of this workspace"}],
                status_code=403
            )

        settings = db.query(WorkspaceSettings).filter_by(workspace_id=workspace_id).first()
        if not settings:
            return format_response(
                status=False,
                message="Settings not found",
                errors=[{"field": "workspace_id", "message": "Workspace settings not found"}],
                status_code=404
            )

        return format_response(
            status=True,
            message="Workspace settings retrieved",
            data={
                "default_voice": settings.default_voice,
                "default_model": settings.default_model,
                "temperature": settings.temperature
            }
        )

    except Exception as e:
        db.rollback()
        return format_response(
            status=False,
            message="Internal Server Error",
            errors=[{"field": "server", "message": str(e)}]
        )

@router.put("/workspaces/{workspace_id}/settings")
def update_workspace_settings(
    workspace_id: int,
    updates: WorkspaceSettingsUpdate = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        workspace = db.query(Workspace).filter_by(id=workspace_id).first()
        if not workspace or workspace.owner_id != current_user.id:
            return format_response(
                status=False,
                message="Only owner can update settings",
                errors=[{"field": "permission", "message": "Only the workspace owner can update settings"}],
                status_code=403
            )

        settings = db.query(WorkspaceSettings).filter_by(workspace_id=workspace_id).first()
        if not settings:
            return format_response(
                status=False,
                message="Settings not found",
                errors=[{"field": "workspace_id", "message": "Workspace settings not found"}],
                status_code=404
            )

        if updates.default_voice is not None:
            settings.default_voice = updates.default_voice
        if updates.default_model is not None:
            settings.default_model = updates.default_model
        if updates.temperature is not None:
            settings.temperature = updates.temperature

        db.commit()

        return format_response(
            status=True,
            message="Settings updated"
        )

    except Exception as e:
        db.rollback()
        return format_response(
            status=False,
            message="Internal Server Error",
            errors=[{"field": "server", "message": str(e)}]
        )
        
@router.get("/list-knowledge-bases")
def list_knowledge_bases(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        # Join KnowledgeBase -> Workspace -> WorkspaceMember
        knowledge_bases = (
            db.query(KnowledgeBase)
            .join(Workspace, KnowledgeBase.workspace)  # join workspace from KB
            .outerjoin(WorkspaceMember, Workspace.id == WorkspaceMember.workspace_id)
            .filter(
                (Workspace.owner_id == current_user.id) |
                (WorkspaceMember.user_id == current_user.id)
            )
            .all()
        )

        result = []

        for kb in knowledge_bases:
            files = db.query(KnowledgeFile).filter_by(kb_id=kb.id).all()
            file_sources = []
            for file in files:
                file_sources.append({
                    "type": "document",
                    "source_id": str(file.id),
                    "filename": file.filename,
                    "file_url": file.file_url 
                })

            result.append({
                "knowledge_base_id": f"knowledge_base_{kb.id}",
                "knowledge_base_name": kb.name,
                "status": getattr(kb, "status", "in_progress"),
                "knowledge_base_sources": file_sources,
                "enable_auto_refresh": kb.enable_auto_refresh,
                "last_refreshed_timestamp": int(kb.last_refreshed.timestamp() * 1000) if kb.last_refreshed else 0
            })

        return format_response(
            status=True,
            message="Knowledge bases retrieved successfully",
            data=result
        )

    except Exception as e:
        db.rollback()
        return format_response(
            status=False,
            message="Internal Server Error",
            errors=[{"field": "server", "message": str(e)}],
            status_code=500
        )
        
#  API KEYS      
@router.post("/workspaces/{workspace_id}/api-keys")
def create_api_key(
    workspace_id: int,
    payload: dict = Body(...),  # expecting: { "name": "my key name" }
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if not is_user_in_workspace(current_user.id, workspace_id, db):
        raise HTTPException(status_code=403, detail="You do not have access to this workspace")

    workspace = db.query(Workspace).filter_by(id=workspace_id).first()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")

    key_value = generate_api_key()
    api_key = APIKey(
        workspace_id=workspace.id,
        name=payload["name"],
        key_value=key_value,
        created_by=current_user.id
    )
    db.add(api_key)
    db.commit()
    db.refresh(api_key)

    return format_response(
    status=True,
    message="API Key created successfully",
    data={
        "id": api_key.id,
        "name": api_key.name,
        "key_value": api_key.key_value,
        "last_edited_by": current_user.username,
        "updated_at": api_key.updated_at.isoformat()
    }
)

@router.get("/workspaces/{workspace_id}/api-keys")
def list_api_keys(
    workspace_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if not is_user_in_workspace(current_user.id, workspace_id, db):
        raise HTTPException(status_code=403, detail="You do not have access to this workspace")

    keys = db.query(APIKey).filter_by(workspace_id=workspace_id).all()
    data = [{
        "id": key.id,
        "name": key.name,
        "key_value": key.key_value,
        "last_edited_by": current_user.username,
        "updated_at": key.updated_at.isoformat(),
        "is_webhook_key": key.is_webhook_key
    } for key in keys]

    return format_response(
        status=True,
        message="API keys fetched successfully",
        data=data
    )

@router.delete("/workspaces/{workspace_id}/api-keys/{key_id}")
def delete_api_key(
    workspace_id: int,
    key_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Check if the user is authorized in this workspace
    if not is_user_in_workspace(current_user.id, workspace_id, db):
        raise HTTPException(status_code=403, detail="You do not have access to this workspace")

    # Fetch the specific API key under that workspace
    key = db.query(APIKey).filter_by(id=key_id, workspace_id=workspace_id).first()
    if not key:
        raise HTTPException(status_code=404, detail="API key not found in this workspace")

    if key.is_webhook_key:
        raise HTTPException(status_code=403, detail="Cannot delete webhook key. Create a new key and set as webhook key first before deleting this key.")

    db.delete(key)
    db.commit()

    return format_response(
        status=True,
        message="API key deleted successfully"
    )

@router.put("/api-keys/{key_id}/rename")
def update_api_key_name(
    key_id: int,
    name: str = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    key = db.query(APIKey).filter_by(id=key_id).first()
    if not key:
        raise HTTPException(status_code=404, detail="API key not found")

    key.name = name
    key.updated_at = datetime.utcnow()
    key.created_by = current_user.id
    db.commit()
    return {"message": "API key renamed"}

@router.put("/api-keys/{key_id}/set-webhook")
def set_webhook_api_key(
    key_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    key = db.query(APIKey).filter_by(id=key_id).first()
    if not key:
        raise HTTPException(status_code=404, detail="API key not found")

    # Clear existing webhook key
    db.query(APIKey).filter(
        APIKey.workspace_id == key.workspace_id,
        APIKey.is_webhook_key == True
    ).update({"is_webhook_key": False})

    key.is_webhook_key = True
    db.commit()
    return {"message": "Webhook API key set"}

__all__ = ['router']