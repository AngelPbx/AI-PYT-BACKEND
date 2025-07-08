import os, shutil, fitz
from docx import Document  #pip install pymupdf python-docx
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from openai import OpenAI
from datetime import timedelta
from fastapi import Depends, Query, HTTPException, Form, UploadFile, File, Body, APIRouter
from utils.helpers import is_user_in_workspace, generate_api_key
from sqlalchemy.orm import Session
import uuid 
import requests
from bs4 import BeautifulSoup
import time,json
from livekit import api
from typing import List
from sqlalchemy.orm import sessionmaker, Session
from db.database import get_db, get_current_user
from models.models import ( User, Workspace, WorkspaceSettings, WorkspaceMember, KnowledgeBase, 
                           KnowledgeFile, APIKey, FileStatus, SourceStatus, pbx_ai_agent, PBXLLM, 
                           ChatSession, LLMVoice, ImportedPhoneNumber
                           )
from models.schemas import (
    UserSignup, UserLogin, UpdateUser,DispatchRequest,CreateRoomRequestSchema,
    WorkspaceCreate, WorkspaceOut, InviteMember,WorkspaceSettingsUpdate, KnowledgeBaseCreate, 
    AgentCreate, AgentOut, PBXLLMCreate, PBXLLMOut, CreateChatRequest, CreateChatResponse,
    VoiceOut, VoiceCreate, PhoneNumberCreate, PhoneNumberOut, APIResponse
)
from utils.helpers import (
    format_response, validate_email,
    validate_password, format_datetime_ist
)
from utils.security import (
    hash_password, verify_password,
    create_token
)


from db.database import engine

from starlette.requests import Request
from starlette.config import Config

router = APIRouter()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

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
        db_user = db.query(User).filter(User.email == user.email).first()
        if not db_user or not verify_password(user.password, db_user.hashed_password):
            raise HTTPException(status_code=401, detail="Invalid credentials")

        token = create_token(data={"email": db_user.email})

        return format_response(
            status=True,
            message="Login successful",
            data={
                "token": 'Bearer ' + token
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


# Helper to format responses

@router.post("/create-room-token")
async def create_room_and_token(
    request: CreateRoomRequestSchema,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)  # get the logged-in user
):
    try:
        LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
        LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
        LIVEKIT_URL = os.getenv("LIVEKIT_URL")

        if not (LIVEKIT_API_KEY and LIVEKIT_API_SECRET and LIVEKIT_URL):
            raise HTTPException(status_code=500, detail="LiveKit credentials not configured")

        workspace_ids = [
            membership.workspace_id for membership in current_user.memberships
        ]

        if not workspace_ids:
            raise HTTPException(status_code=403, detail="User is not a member of any workspace")

        # 🔥 Step 2: Search for agent in user's workspaces
        agent = (
            db.query(pbx_ai_agent)
            .filter(
                pbx_ai_agent.name == request.agent_name,
                pbx_ai_agent.workspace_id.in_(workspace_ids)
            )
            .first()
        )
        pbxllm = (
            db.query(PBXLLM)
            .filter(PBXLLM.workspace_id == agent.workspace_id)
            .first()
        )
        print(f"Agent found: {agent is not None}, PBXLLM found: {pbxllm is not None}")

        if not agent and pbxllm:
            raise HTTPException(status_code=404, detail="No LLM configuration & Agent not found for this user")
        

        # ✅ Print fetched agent data in console
        print("Fetched Agent Data:")
        # print({
        #     "id": agent.id,
        #     "name": agent.name,
        #     "voice_id": agent.voice_id,
        #     "voice_model": agent.voice_model,
        #     "language": agent.language,
        #     "response_engine": agent.response_engine,
        #     "ambient_sound": agent.ambient_sound
        # })

        # 🔥 Step 2: Create room (optional - LiveKit auto-creates)
        lkapi = api.LiveKitAPI(
            url=LIVEKIT_URL,
            api_key=LIVEKIT_API_KEY,
            api_secret=LIVEKIT_API_SECRET
        )
        try:
            lkapi.room.create_room(
                api.CreateRoomRequest(
                    name=request.room_name,
                    empty_timeout=10 * 60,  # 10 minutes timeout
                    max_participants=10,
                    metadata=json.dumps(request.metadata) if request.metadata else None
                )
            )
            print(f"Room '{request.room_name}' created successfully.")
        except Exception as e:
            if "already exists" not in str(e):
                raise HTTPException(status_code=500, detail=f"LiveKit Error: {e}")

        # 🔥 Step 3: Generate token
        token = api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET) \
            .with_identity(request.participant_identity) \
            .with_name(request.participant_name) \
            .with_grants(api.VideoGrants(
                room_join=True,
                room=request.room_name,
            ))
        jwt_token = token.to_jwt()
        return jwt_token

        # return format_response(
        #     status=True,
        #     message="Room and token created successfully",
        #     data={
        #         "room": request.room_name,
        #         "token": "Bearer " + jwt_token,
        #         "metadata": request.metadata,
        #         "agent": {
        #             "id": agent.id,
        #             "name": agent.name,
        #             "voice_model": agent.voice_model,
        #             "language": agent.language,
        #             "response_engine": agent.response_engine
        #         }
        #     }
        # )

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@router.post("/create-dispatch")
async def create_dispatch(request: DispatchRequest):
    lkapi = api.LiveKitAPI()
    try:
        dispatch = await lkapi.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name=request.agent_name,
                room=request.room_name,
                metadata=json.dumps(request.metadata)
            )
        )
        return {
            "status": "success",
            "dispatch_id": dispatch.id,
            "room": dispatch.room,
            "agent_name": dispatch.agent_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await lkapi.aclose()    

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

config = Config(".env")

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

# Agent apis--------------------------------------------------

@router.post("/agents", response_model=AgentOut)
def create_agent(
    payload: AgentCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if not is_user_in_workspace(current_user.id, payload.workspace_id, db):
        pass
        # raise HTTPException(status_code=403, detail="You do not have access to this workspace")

    agent = pbx_ai_agent(
        workspace_id=payload.workspace_id,
        name=payload.name,
        voice_id=payload.voice_id,
        voice_model=payload.voice_model,
        fallback_voice_ids=payload.fallback_voice_ids,
        voice_temperature=payload.voice_temperature,
        voice_speed=payload.voice_speed,
        volume=payload.volume,
        responsiveness=payload.responsiveness,
        interruption_sensitivity=payload.interruption_sensitivity,
        enable_backchannel=payload.enable_backchannel,
        backchannel_frequency=payload.backchannel_frequency,
        backchannel_words=payload.backchannel_words,
        reminder_trigger_ms=payload.reminder_trigger_ms,
        reminder_max_count=payload.reminder_max_count,
        ambient_sound=payload.ambient_sound,
        ambient_sound_volume=payload.ambient_sound_volume,
        language=payload.language,
        webhook_url=payload.webhook_url,
        boosted_keywords=payload.boosted_keywords,
        opt_out_sensitive_data_storage=payload.opt_out_sensitive_data_storage,
        opt_in_signed_url=payload.opt_in_signed_url,
        pronunciation_dictionary=[
            p.model_dump() for p in payload.pronunciation_dictionary
        ] if payload.pronunciation_dictionary else None,
        normalize_for_speech=payload.normalize_for_speech,
        end_call_after_silence_ms=payload.end_call_after_silence_ms,
        max_call_duration_ms=payload.max_call_duration_ms,
        voicemail_option=payload.voicemail_option.model_dump() if payload.voicemail_option else None,
        post_call_analysis_data=[
            a.model_dump() for a in payload.post_call_analysis_data
        ] if payload.post_call_analysis_data else None,
        post_call_analysis_model=payload.post_call_analysis_model,
        begin_message_delay_ms=payload.begin_message_delay_ms,
        ring_duration_ms=payload.ring_duration_ms,
        stt_mode=payload.stt_mode,
        vocab_specialization=payload.vocab_specialization,
        allow_user_dtmf=payload.allow_user_dtmf,
        user_dtmf_options=payload.user_dtmf_options.model_dump() if payload.user_dtmf_options else None,
        denoising_mode=payload.denoising_mode,
        response_engine=payload.response_engine.model_dump(),
        version=payload.version,
        last_modification_timestamp=int(time.time() * 1000),
    )



    db.add(agent)
    db.commit()
    db.refresh(agent)
    return agent


# @router.get("/all-agents/{workspace_id}", response_model=List[AgentOut])
# def list_agents(
#     workspace_id: int,
#     db: Session = Depends(get_db),
#     current_user: User = Depends(get_current_user)
# ):
#     if not is_user_in_workspace(current_user.id, workspace_id, db):
#         raise HTTPException(status_code=403, detail="You do not have access to this workspace")

#     agents = db.query(pbx_ai_agent).filter(pbx_ai_agent.workspace_id == workspace_id).all()
#     # print(db.query(pbx_ai_agent).all())
#     return agents

@router.get("/all-agents/{workspace_id}")
def list_my_agents(
    workspace_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if not is_user_in_workspace(current_user.id, workspace_id, db):
        raise HTTPException(status_code=403, detail="You do not have access to this workspace")
    try:
        # Get all workspace IDs the user is a member of
        workspace_ids = db.query(WorkspaceMember.workspace_id).filter(
            WorkspaceMember.user_id == current_user.id
        ).subquery()

        # Fetch all agents in those workspaces
        agents = db.query(pbx_ai_agent).filter(
            pbx_ai_agent.workspace_id.in_(workspace_ids)
        ).all()

        # Serialize agents into your desired format
        agent_list = []
        for agent in agents:
            agent_list.append({
                "agent_id": agent.id,
                "version": agent.version,
                "is_published": agent.is_published,
                "response_engine": agent.response_engine or {},
                "agent_name": agent.name,
                "voice_id": agent.voice_id,
                "voice_model": agent.voice_model,
                "fallback_voice_ids": agent.fallback_voice_ids or [],
                "voice_temperature": agent.voice_temperature,
                "voice_speed": agent.voice_speed,
                "volume": agent.volume,
                "responsiveness": agent.responsiveness,
                "interruption_sensitivity": agent.interruption_sensitivity,
                "enable_backchannel": agent.enable_backchannel,
                "backchannel_frequency": agent.backchannel_frequency,
                "backchannel_words": agent.backchannel_words or [],
                "reminder_trigger_ms": agent.reminder_trigger_ms,
                "reminder_max_count": agent.reminder_max_count,
                "ambient_sound": agent.ambient_sound,
                "ambient_sound_volume": agent.ambient_sound_volume,
                "language": agent.language,
                "webhook_url": agent.webhook_url,
                "boosted_keywords": agent.boosted_keywords or [],
                "opt_out_sensitive_data_storage": agent.opt_out_sensitive_data_storage,
                "opt_in_signed_url": agent.opt_in_signed_url,
                "pronunciation_dictionary": agent.pronunciation_dictionary or [],
                "normalize_for_speech": agent.normalize_for_speech,
                "end_call_after_silence_ms": agent.end_call_after_silence_ms,
                "max_call_duration_ms": agent.max_call_duration_ms,
                "voicemail_option": agent.voicemail_option or {},
                "post_call_analysis_data": agent.post_call_analysis_data or [],
                "post_call_analysis_model": agent.post_call_analysis_model,
                "begin_message_delay_ms": agent.begin_message_delay_ms,
                "ring_duration_ms": agent.ring_duration_ms,
                "stt_mode": agent.stt_mode,
                "vocab_specialization": agent.vocab_specialization,
                "allow_user_dtmf": agent.allow_user_dtmf,
                "user_dtmf_options": agent.user_dtmf_options or {},
                "denoising_mode": agent.denoising_mode,
                "last_modification_timestamp": agent.last_modification_timestamp
            })

        return format_response(
            status=True,
            message="Agents fetched successfully",
            data=agent_list
        )

    except Exception as e:
        db.rollback()
        return format_response(
            status=False,
            message=f"Internal Server Error: {str(e)}",
            data=None
        )


@router.get("/agents/{agent_id}", response_model=AgentOut)
def get_agent(
    agent_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Fetch the agent by ID
    agent = db.query(pbx_ai_agent).filter(pbx_ai_agent.id == agent_id).first()

    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Optional: Check workspace access if needed
    if not is_user_in_workspace(current_user.id, agent.workspace_id, db):
        raise HTTPException(status_code=403, detail="Access denied to this workspace")

    return agent


# PBX LLM APIs--------------------------------------------------

@router.post("/pbx-llms", response_model=PBXLLMOut)
def create_pbx_llm(
    payload: PBXLLMCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Validate workspace access
    if not is_user_in_workspace(current_user.id, payload.workspace_id, db):
        raise HTTPException(status_code=403, detail="You do not have access to this workspace")

    workspace = db.query(Workspace).filter(Workspace.id == payload.workspace_id).first()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")

    llm = PBXLLM(
        workspace_id=payload.workspace_id,
        version=payload.version,
        model=payload.model,
        s2s_model=payload.s2s_model,
        model_temperature=payload.model_temperature,
        model_high_priority=payload.model_high_priority,
        tool_call_strict_mode=payload.tool_call_strict_mode,
        general_prompt=payload.general_prompt,
        general_tools=payload.general_tools,
        states=payload.states,
        starting_state=payload.starting_state,
        begin_message=payload.begin_message,
        default_dynamic_variables=payload.default_dynamic_variables,
        knowledge_base_ids=payload.knowledge_base_ids,
        last_modification_timestamp=int(time.time() * 1000)
    )
    db.add(llm)
    db.commit()
    db.refresh(llm)
   
    return PBXLLMOut.model_validate(llm)

@router.get("/pbx-llms", response_model=PBXLLMOut)
def get_pbx_llm(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)  # Kept for consistency, though not used
):
    # Hardcoded data
    llm_data = {
        "workspace_id": 1,
        "version": 1,
        "model": "gpt-4o",
        "s2s_model": "whisper-large-v3",
        "model_temperature": 0.7,
        "model_high_priority": True,
        "tool_call_strict_mode": False,
        "general_prompt": "## Identity\nYou are Kate from the appointment department at Webvio Health calling Cindy over the phone to prepare for the annual checkup coming up. You are a pleasant and friendly receptionist caring deeply for the user. You don't provide medical advice but would use the medical knowledge to understand user responses.\n\n## Style Guardrails\nBe Concise: Respond succinctly, addressing one topic at most.\nEmbrace Variety: Use diverse language and rephrasing to enhance clarity without repeating content.\nBe Conversational: Use everyday language, making the chat feel like talking to a friend.\nBe Proactive: Lead the conversation, often wrapping up with a question or next-step suggestion.\nAvoid multiple questions in a single response.\nGet clarity: If the user only partially answers a question, or if the answer is unclear, keep asking to get clarity.\nUse a colloquial way of referring to the date (like Friday, January 14th, or Tuesday, January 12th, 2024 at 8am).\n\n## Response Guideline\nAdapt and Guess: Try to understand transcripts that may contain transcription errors. Avoid mentioning \"transcription error\" in the response.",
        "general_tools": [
            {
                "type": "end_call",
                "name": "end_call",
                "description": "Hang up the call"
            }
        ],
        "begin_message": "Connecting you with the assistant...",
        "default_dynamic_variables": {
            "company_name": "VoiceAI",
            "timezone": "UTC+5:30"
        },
        "knowledge_base_ids": []
    }
    
    # Validate and return the data using PBXLLMOut
    return PBXLLMOut.model_validate(llm_data)


@router.get("/all-pbx-llms/{workspace_id}", response_model=List[PBXLLMOut])
def list_pbx_llms(
    workspace_id=int,
    db: Session = Depends(get_db)
):
    llms = db.query(PBXLLM).filter(PBXLLM.workspace_id == workspace_id).all()
    return [PBXLLMOut.model_validate(llm) for llm in llms]

# Chat room APIs--------------------------------------------------
@router.post("/create-chat", response_model=CreateChatResponse)
def create_chat(payload: CreateChatRequest, db: Session = Depends(get_db)):
    try:
        chat = ChatSession(
            agent_id=payload.agent_id,
            agent_version=payload.agent_version or 0,
            chat_status=payload.chat_status,
            retell_llm_dynamic_variables=payload.retell_llm_dynamic_variables,
            collected_dynamic_variables={},
            chat_metadata=payload.metadata,
            start_timestamp=payload.start_timestamp,
            end_timestamp=payload.end_timestamp,
            transcript=payload.transcript or "null",
            message_with_tool_calls=[],  # ensure it's a list of dicts or JSON-serializable
            chat_cost=payload.chat_cost.model_dump() if payload.chat_cost else None,
            chat_analysis=payload.chat_analysis.model_dump() if payload.chat_analysis else None
        )

        db.add(chat)
        db.commit()
        db.refresh(chat)

        return CreateChatResponse(
            chat_id=chat.chat_id,
            agent_id=chat.agent_id,
            chat_status=chat.chat_status,
            retell_llm_dynamic_variables=chat.retell_llm_dynamic_variables,
            collected_dynamic_variables=chat.collected_dynamic_variables,
            start_timestamp=chat.start_timestamp,
            end_timestamp=chat.end_timestamp,
            transcript=chat.transcript,
            message_with_tool_calls=chat.message_with_tool_calls,
            metadata=chat.chat_metadata,
            chat_cost=chat.chat_cost,
            chat_analysis=chat.chat_analysis
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# voice
@router.post("/llm-voice", response_model=VoiceOut)
def create_voice(
    payload: VoiceCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)  # optional if you want auth
):
    existing = db.query(LLMVoice).filter_by(voice_id=payload.voice_id).first()
    if existing:
        raise HTTPException(status_code=400, detail="Voice ID already exists")
    
    voice = LLMVoice(**payload.dict())
    db.add(voice)
    db.commit()
    db.refresh(voice)
    return voice


@router.get("/llm-voice/{voice_id}", response_model=VoiceOut)
def get_voice(
    voice_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)  # optional
):
    voice = db.query(LLMVoice).filter_by(voice_id=voice_id).first()
    if not voice:
        raise HTTPException(status_code=404, detail="Voice not found")
    return voice


# @router.get("/all-voices", response_model=List[VoiceOut])
# def list_voices(
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     try:
#         voices_data = [
#             {
#                 "voice_id": "openai-shimmer",
#                 "voice_name": "Shimmer",
#                 "provider": "elevenlabs",
#                 "gender": "female",
#                 "accent": "Neutral",
#                 "age": "Adult",
#                 "preview_audio_url": "https://example.com/previews/shimmer.mp3"
#             },
#             {
#                 "voice_id": "elevn-echo",
#                 "voice_name": "Echo",
#                 "provider": "elevenlabs",
#                 "gender": "male",
#                 "accent": "British",
#                 "age": "Middle-aged",
#                 "preview_audio_url": "https://example.com/previews/echo.mp3"
#             },
#             {
#                 "voice_id": "elevn-nova",
#                 "voice_name": "Nova",
#                 "provider": "elevenlabs",
#                 "gender": "female",
#                 "accent": "American",
#                 "age": "Young Adult",
#                 "preview_audio_url": "https://example.com/previews/nova.mp3"
#             },
#             {
#                 "voice_id": "elevn-pulse",
#                 "voice_name": "Pulse",
#                 "provider": "elevenlabs",
#                 "gender": "male",
#                 "accent": "Australian",
#                 "age": "Adult",
#                 "preview_audio_url": "https://example.com/previews/pulse.mp3"
#             }
#         ]
#         # Convert to VoiceOut objects and return as a list
#         return [VoiceOut.model_validate(voice) for voice in voices_data]
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail={
#                 "status": False,
#                 "message": "Internal Server Error",
#                 "errors": [{"field": "server", "message": str(e)}]
#             }
#         )

@router.get("/all-voices", response_model=APIResponse)
def list_voices(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        voices_data = [
            {
                "voice_id": "openai-shimmer",
                "voice_name": "Shimmer",
                "provider": "elevenlabs",
                "gender": "female",
                "accent": "Neutral",
                "age": "Adult",
                "preview_audio_url": "https://example.com/previews/shimmer.mp3"
            },
            {
                "voice_id": "elevn-echo",
                "voice_name": "Echo",
                "provider": "elevenlabs",
                "gender": "male",
                "accent": "British",
                "age": "Middle-aged",
                "preview_audio_url": "https://example.com/previews/echo.mp3"
            },
            {
                "voice_id": "elevn-nova",
                "voice_name": "Nova",
                "provider": "elevenlabs",
                "gender": "female",
                "accent": "American",
                "age": "Young Adult",
                "preview_audio_url": "https://example.com/previews/nova.mp3"
            },
            {
                "voice_id": "elevn-pulse",
                "voice_name": "Pulse",
                "provider": "elevenlabs",
                "gender": "male",
                "accent": "Australian",
                "age": "Adult",
                "preview_audio_url": "https://example.com/previews/pulse.mp3"
            }
        ]
        
        # Validate and convert to VoiceOut objects
        try:
            voices = [VoiceOut.model_validate(voice) for voice in voices_data]
        except ValueError as validation_error:
            return APIResponse(
                status=False,
                message="Validation error",
                data=None,
                errors=[
                    {
                        "field": "voice_data",
                        "message": str(validation_error)
                    }
                ]
            )
        
        # Return successful response
        return APIResponse(
            status=True,
            message="Voices fetched successfully",
            data=voices,
            errors=[]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": False,
                "message": "Internal Server Error",
                "data": None,
                "errors": [
                    {
                        "field": "server",
                        "message": str(e)
                    }
                ]
            }
        )

#import phone
@router.post("/import-phone-number", response_model=PhoneNumberOut, status_code=201)
def import_phone_number(
    payload: PhoneNumberCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)  # optional
):
    existing = db.query(ImportedPhoneNumber).filter_by(phone_number=payload.phone_number).first()
    if existing:
        raise HTTPException(status_code=400, detail="Phone number already exists")

    area_code = int(payload.phone_number[2:5])
    pretty = f"+1 ({payload.phone_number[2:5]}) {payload.phone_number[5:8]}-{payload.phone_number[8:]}"
    timestamp = int(time.time() * 1000)

    phone = ImportedPhoneNumber(
        phone_number=payload.phone_number,
        phone_number_type=payload.phone_number_type,
        phone_number_pretty=pretty,
        inbound_agent_id=payload.inbound_agent_id,
        outbound_agent_id=payload.outbound_agent_id,
        inbound_agent_version=payload.inbound_agent_version,
        outbound_agent_version=payload.outbound_agent_version,
        area_code=area_code,
        nickname=payload.nickname,
        inbound_webhook_url=payload.inbound_webhook_url,
        last_modification_timestamp=timestamp,
    )

    db.add(phone)
    db.commit()
    db.refresh(phone)
    return phone

@router.get("/import-phone-number/{phone_number}", response_model=PhoneNumberOut)
def get_phone_number(
    phone_number: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)  # optional
):
    phone = db.query(ImportedPhoneNumber).filter_by(phone_number=phone_number).first()
    if not phone:
        raise HTTPException(status_code=404, detail="Phone number not found")
    return phone

@router.get("/all-import-phone-number", response_model=List[PhoneNumberOut])
def list_phone_numbers(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)  # optional
):
    return db.query(ImportedPhoneNumber).all()

# @router.get("/auth/github/login")
# async def github_login(request: Request):
#     redirect_uri = request.url_for("github_callback")
#     return await oauth.github.authorize_redirect(request, redirect_uri)

# @router.get("/auth/github/callback")
# async def github_callback(request: Request, db: Session = Depends(get_db)):
#     try:
#         token = await oauth.github.authorize_access_token(request)
#         github_user = await oauth.github.get('user', token=token)
#         github_user = github_user.json()

#         username = github_user["login"]
#         email = github_user.get("email")

#         if not email:
#             # GitHub may hide public email, so fetch from email endpoint
#             emails = await oauth.github.get('user/emails', token=token)
#             email_list = emails.json()
#             email = next((item["email"] for item in email_list if item["primary"] and item["verified"]), None)

#         if not email:
#             raise HTTPException(status_code=400, detail="Email not available from GitHub")

#         user = db.query(User).filter(User.email == email).first()

#         if not user:
#             # Create new user
#             user = User(
#                 username=username,
#                 email=email,
#                 full_name=github_user.get("name") or username,
#                 hashed_password="",
#                 created_at=datetime.utcnow(),
#                 updated_at=datetime.utcnow()
#             )
#             db.add(user)
#             db.commit()
#             db.refresh(user)

#             # Create workspace and membership
#             workspace = Workspace(
#                 name=f"{username}_workspace",
#                 description="Default workspace",
#                 owner_id=user.id,
#                 created_at=datetime.utcnow()
#             )
#             db.add(workspace)
#             db.commit()
#             db.refresh(workspace)

#             settings = WorkspaceSettings(
#                 workspace_id=workspace.id,
#                 default_model="gpt-4",
#                 default_voice="echo",
#                 temperature=1
#             )
#             member = WorkspaceMember(
#                 user_id=user.id,
#                 workspace_id=workspace.id,
#                 role="owner"
#             )
#             db.add_all([settings, member])
#             db.commit()

#         # Issue JWT token
#         token, expire = create_token({
#             "username": user.username,
#             "expire_minutes": os.getenv("TOKEN_EXPIRE_MINUTES", 60)
#         })

#         return format_response(
#             status=True,
#             message="GitHub login successful",
#             data={
#                 "token": "Bearer " + token,
#                 "expires_at": expire.isoformat() + "Z",
#                 "expire_duration_minutes": int(os.getenv("TOKEN_EXPIRE_MINUTES", 60)),
#             }
#         )

#     except Exception as e:
#         db.rollback()
#         return format_response(
#             status=False,
#             message="GitHub login failed",
#             errors=[{"field": "server", "message": str(e)}]
#         )

@router.get("/list-phone-numbers")
def list_phone_numbers(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        phone_numbers_data = [
            {
                "id": "pn_001",
                "phone_number": "+14157774444",
                "phone_number_type": "retell-twilio",
                "phone_number_pretty": "+1 (415) 777-4444",
                "inbound_agent_id": "oBeDLoLOeuAbiuaMFXRtDOLriTJ5tSxD",
                "outbound_agent_id": "oBeDLoLOeuAbiuaMFXRtDOLriTJ5tSxD",
                "inbound_agent_version": "v1.0",
                "outbound_agent_version": "v1.0",
                "area_code": "415",
                "nickname": "Main Line",
                "inbound_webhook_url": "https://api.example.com/webhooks/inbound/main",
                "last_modification_timestamp": "2025-07-07T18:45:00+05:30"
            },
            {
                "id": "pn_002",
                "phone_number": "+12025550123",
                "phone_number_type": "retell-twilio",
                "phone_number_pretty": "+1 (202) 555-0123",
                "inbound_agent_id": "xYcZMnPQrsTuvWxyZABcDEFghIJkLmNo",
                "outbound_agent_id": "xYcZMnPQrsTuvWxyZABcDEFghIJkLmNo",
                "inbound_agent_version": "v1.1",
                "outbound_agent_version": "v1.1",
                "area_code": "202",
                "nickname": "Support Line",
                "inbound_webhook_url": "https://api.example.com/webhooks/inbound/support",
                "last_modification_timestamp": "2025-07-06T12:30:00+05:30"
            },
            {
                "id": "pn_003",
                "phone_number": "+13105556789",
                "phone_number_type": "retell-twilio",
                "phone_number_pretty": "+1 (310) 555-6789",
                "inbound_agent_id": "pQrStUvWxYzAbCdEfGhIjKlMnOpQrStU",
                "outbound_agent_id": "pQrStUvWxYzAbCdEfGhIjKlMnOpQrStU",
                "inbound_agent_version": "v2.0",
                "outbound_agent_version": "v2.0",
                "area_code": "310",
                "nickname": "Sales Line",
                "inbound_webhook_url": "https://api.example.com/webhooks/inbound/sales",
                "last_modification_timestamp": "2025-07-05T09:15:00+05:30"
            },
            {
                "id": "pn_004",
                "phone_number": "+14405559876",
                "phone_number_type": "retell-twilio",
                "phone_number_pretty": "+1 (440) 555-9876",
                "inbound_agent_id": "aBcDeFgHiJkLmNoPqRsTuVwXyZaBcDeF",
                "outbound_agent_id": "aBcDeFgHiJkLmNoPqRsTuVwXyZaBcDeF",
                "inbound_agent_version": "v1.2",
                "outbound_agent_version": "v1.2",
                "area_code": "440",
                "nickname": "Customer Care",
                "inbound_webhook_url": "https://api.example.com/webhooks/inbound/customer",
                "last_modification_timestamp": "2025-07-04T14:20:00+05:30"
            }
        ]

        # Validate data with PhoneNumberOut schema
        data = [PhoneNumberOut(**phone).dict() for phone in phone_numbers_data]

        return format_response(
            status=True,
            message="Phone numbers retrieved successfully",
            data=data
        )

    except Exception as e:
        return format_response(
            status=False,
            message="Internal Server Error",
            errors=[{"field": "server", "message": str(e)}]
        )