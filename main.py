from fastapi import FastAPI, HTTPException, Depends, Header, UploadFile, File, Form, Body
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import Boolean, ForeignKey, DateTime, Column, Integer, String, create_engine
from sqlalchemy.orm import relationship, sessionmaker, declarative_base, Session
from sqlalchemy.sql import func
from passlib.context import CryptContext
import jwt, re, os
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
import whisper 
from openai import OpenAI
import uuid
from datetime import datetime
from pathlib import Path
import pytz

from fastapi.staticfiles import StaticFiles

from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

load_dotenv()

whisper_model = whisper.load_model("base")
client = OpenAI()

#  CONFIGURATION 
SECRET_KEY = os.getenv("JWT_SECRET", "supersecret")
ALGORITHM = "HS256"
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://myuser:password@localhost:5432/mydb")

#  DB SETUP 
Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

#  USER MODEL 
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False)
    full_name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    retall_api_key = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime, nullable=False)
    updated_at = Column(DateTime, default=datetime, onupdate=datetime, nullable=False)
    is_active = Column(Boolean, default=True)
      
    workspaces = relationship("Workspace", back_populates="owner", cascade="all, delete", passive_deletes=True)
    memberships = relationship("WorkspaceMember", back_populates="user")
    edited_keys = relationship("APIKey", back_populates="user")

# WORKSPACE
class Workspace(Base):
    __tablename__ = "workspaces"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    owner_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    owner = relationship("User", back_populates="workspaces", passive_deletes=True)
    members = relationship("WorkspaceMember", back_populates="workspace", cascade="all, delete")
    knowledge_bases = relationship("KnowledgeBase", back_populates="workspace", cascade="all, delete")
    settings = relationship("WorkspaceSettings", back_populates="workspace", uselist=False, cascade="all, delete")
    api_keys = relationship("APIKey", back_populates="workspace", cascade="all, delete-orphan")

class WorkspaceMember(Base):
    __tablename__ = "workspace_members"
    id = Column(Integer, primary_key=True, index=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    role = Column(String, default="member")

    workspace = relationship("Workspace", back_populates="members")
    user = relationship("User", back_populates="memberships")
    
class WorkspaceSettings(Base):
    __tablename__ = "workspace_settings"

    id = Column(Integer, primary_key=True, index=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id"), unique=True, nullable=False)
    default_voice = Column(String, default="echo")
    default_model = Column(String, default="gpt-4")
    temperature = Column(Integer, default=1)

    workspace = relationship("Workspace", back_populates="settings")

# KNOWLEDGE BASE
class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    workspace_id = Column(Integer, ForeignKey("workspaces.id"), nullable=False)
    enable_auto_refresh = Column(Boolean, default=False, nullable=False)
    auto_refresh_interval = Column(Integer, default=24, nullable=False)  # hours
    last_refreshed = Column(DateTime, nullable=True)
    
    workspace = relationship("Workspace", back_populates="knowledge_bases")

class KnowledgeFile(Base):
    __tablename__ = "knowledge_files"
    id = Column(Integer, primary_key=True, index=True)
    kb_id = Column(Integer, ForeignKey("knowledge_bases.id"), nullable=False)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())

    knowledge_base = relationship("KnowledgeBase")
    
class APIKey(Base):
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id"), nullable=False)
    name = Column(String, nullable=False)
    key_value = Column(String, unique=True, nullable=False)
    is_webhook_key = Column(Boolean, default=False)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    workspace = relationship("Workspace", back_populates="api_keys")
    user = relationship("User", back_populates="edited_keys")

Workspace.api_keys = relationship("APIKey", back_populates="workspace", cascade="all, delete-orphan")
User.edited_keys = relationship("APIKey", back_populates="user")

# Base.metadata.drop_all(bind=engine, checkfirst=True)
Base.metadata.create_all(bind=engine)

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

#  SECURITY 
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(password: str, hashed: str) -> bool:
    return pwd_context.verify(password, hashed)

def create_token(data: dict):
    return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str):
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    payload = decode_token(token)
    user = db.query(User).filter(User.username == payload.get("username")).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

#  SCHEMAS - what is schemas? 
#  Schemas are Pydantic models that define the structure of the data being sent and received in API requests and responses.
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
        orm_mode = True

class InviteMember(BaseModel):
    username: str
    role: Optional[str] = "member"
    
class KnowledgeBaseCreate(BaseModel):
    name: str

class KnowledgeBaseOut(BaseModel):
    id: int
    name: str
    file_path: str
    created_at: datetime
    workspace_id: int

    class Config:
        orm_mode = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class KnowledgeFileOut(BaseModel):
    id: int
    filename: str
    file_path: str
    kb_id: int      
    uploaded_at: datetime
    
    class Config:
        orm_mode = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        
class APIKeyCreate(BaseModel):
    name: str
        
class WorkspaceSettingsUpdate(BaseModel):
    default_voice: Optional[str] = None
    default_model: Optional[str] = None
    temperature: Optional[float] = None

#  VALIDATORS 
def validate_email(email: str):
    if not re.match(r"^[^@]+@[^@]+\.(com)$", email):
        raise HTTPException(status_code=400, detail="Invalid email format")

def validate_password(password: str):
    if not re.match(r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[\W_]).{8,}$", password):
        raise HTTPException(status_code=400, detail="Password too weak")
    

def format_response(status: bool, message: str, data: Optional[Any] = None, errors: Optional[List[Dict[str, str]]] = None, status_code: int = 201) -> JSONResponse:
    response = {
        "status": status,
        "message": message,
        "data": data
    }

    if errors is not None:
        response["errors"] = errors

    return JSONResponse(
        content=response,
        status_code=status_code
    )

#  FASTAPI APP 
app = FastAPI()

app.mount("/files", StaticFiles(directory="uploads"), name="files")

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    errors = []
    for err in exc.errors():
        loc_parts = [str(part) for part in err["loc"] if part != "body"]
        field = ".".join(loc_parts)
        errors.append({
            "field": field,
            "message": err["msg"]
        })

    return JSONResponse(
        status_code=422,
        content=format_response(
            status=False,
            message="Validation error",
            errors=errors
        )
    )

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": False,
            "message": exc.detail,
            "data": None
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content=format_response(
            status=False,
            message="Internal server error",
            errors=[{"field": "server", "message": str(exc)}]
        )
    )

#  ROUTES 
@app.post("/signup")
def signup(user: UserSignup, db: Session = Depends(get_db)):
    try:
        # Step 1: Validate inputs
        validate_email(user.email)
        validate_password(user.password)

        # Step 2: Check if user already exists
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

        # Step 3: Create user
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

        # Step 4: Create default workspace
        workspace = Workspace(
            name=f"{user.username}_workspace",
            description="Default workspace",
            owner_id=new_user.id,
            created_at=datetime.utcnow()
        )
        db.add(workspace)
        db.commit()
        db.refresh(workspace)

        # Step 5: Add user as member and create workspace settings
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

@app.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    try:
        db_user = db.query(User).filter(User.username == user.username).first()
        if not db_user or not verify_password(user.password, db_user.hashed_password):
            raise HTTPException(status_code=401, detail="Invalid credentials")

        token = create_token({"username": db_user.username})
        return {"status": True, "message": "Login successful", "token": 'Bearer_' + token}
    
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.put("/user/update")
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
            except ValueError as e:
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
            except ValueError as e:
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

        db.commit()

        return format_response(
            status=True,
            message="User details updated successfully",
            data={
                
            }
        )

    except Exception as e:
        db.rollback()
        return format_response(
            status=False,
            message="Internal Server Error",
            errors=[{"field": "server", "message": str(e)}]
        )

@app.get("/user/status")
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

@app.post("/ai-agent")
async def ai_agent(
    audio: UploadFile = File(...),
    workspace_id: int = Form(...),
    authorization: str = Header(...),
    db: Session = Depends(get_db)
):
    try:
        if not authorization.startswith("Bearer "):
            return format_response(
                status=False,
                message="Invalid token format",
                errors=[{"field": "authorization", "message": "Invalid token format"}]
            )

        token = authorization.split(" ")[1]
        payload = decode_token(token)
        username = payload.get("username")

        user = db.query(User).filter_by(username=username).first()
        print(f"Authenticated user: {user.username}, ID: {user.id}")
        if not user:
            return format_response(
                status=False,
                message="User not found",
                errors=[{"field": "user", "message": "User not found"}]
            )

        membership = db.query(WorkspaceMember).filter_by(workspace_id=workspace_id, user_id=user.id).first()
        if not membership:
            return format_response(
                status=False,
                message="Access denied to workspace",
                errors=[{"field": "workspace", "message": "Access denied"}]
            )

        settings = db.query(WorkspaceSettings).filter_by(workspace_id=workspace_id).first()
        if not settings:
            return format_response(
                status=False,
                message="Workspace settings not found",
                errors=[{"field": "workspace_settings", "message": "Settings not found"}]
            )

        with open("temp.wav", "wb") as f:
            f.write(await audio.read())

        transcript = whisper_model.transcribe("temp.wav")["text"]

        ai_reply = client.chat.completions.create(
            model=settings.default_model,
            messages=[{"role": "user", "content": transcript}],
            temperature=settings.temperature
        ).choices[0].message.content.strip()

        speech = client.audio.speech.create(
            model="tts-1",
            voice=settings.default_voice,
            input=ai_reply
        )

        output_file = f"output_{uuid.uuid4()}.mp3"
        speech.stream_to_file(output_file)

        return format_response(
            status=True,
            message="AI agent response generated",
            data={
                "user": username,
                "transcript": transcript,
                "reply": ai_reply
            }
        )

    except Exception as e:
        return format_response(
            status=False,
            message="Internal Server Error",
            errors=[{"field": "server", "message": str(e)}]
        )
    
@app.post("/workspaces")
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

@app.get("/workspaces")
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
        db.rollback()
        return format_response(
            status=False,
            message="Internal Server Error",
            errors=[{"field": "server", "message": str(e)}]
        )

@app.post("/workspaces/{workspace_id}/invite")
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

@app.delete("/workspaces/{workspace_id}")
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

UPLOAD_DIR = Path("uploads").resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@app.post("/workspaces/{workspace_id}/knowledge-bases")
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

@app.get("/workspaces/{workspace_id}/knowledge-bases")
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
        
@app.get("/knowledge-bases/{kb_id}")
def get_knowledge_base(
    kb_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        kb = db.query(KnowledgeBase).filter_by(id=kb_id).first()
        if not kb:
            return format_response(
                status=False,
                message="Knowledge base not found",
                errors=[{"field": "kb_id", "message": "Knowledge base not found"}],
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
            "knowledge_base_id": f"knowledge_base_{kb.id}",
            "knowledge_base_name": kb.name,
            "status": "ready",
            "knowledge_base_sources": sources,
            "enable_auto_refresh": kb.enable_auto_refresh,
            "last_refreshed_timestamp": int(kb.last_refreshed.timestamp() * 1000)
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
            errors=[{"field": "server", "message": str(e)}]
        )

@app.post("/knowledge-bases/{kb_id}/sources")
async def add_kb_source(
    kb_id: int,
    source_type: str = Form(...),
    file: UploadFile = File(None),
    url: str = Form(None),
    text_filename: str = Form(None),
    text_content: str = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        kb = db.query(KnowledgeBase).filter_by(id=kb_id).first()
        if not kb:
            return format_response(
                status=False,
                message="Knowledge base not found",
                errors=[{"field": "kb_id", "message": "Knowledge base not found"}],
                status_code=404
            )

        membership = db.query(WorkspaceMember).filter_by(workspace_id=kb.workspace_id, user_id=current_user.id).first()
        if not membership:
            return format_response(
                status=False,
                message="Access denied",
                errors=[{"field": "workspace_id", "message": "User is not a member of this workspace"}],
                status_code=403
            )

        kb_path = UPLOAD_DIR / f"workspace_{kb.workspace_id}" / f"knowledge_base_{kb_id}"
        kb_path.mkdir(parents=True, exist_ok=True)

        file_path = None
        file_name = None
        source_details = {}

        if source_type == "file":
            if not file:
                return format_response(
                    status=False,
                    message="No file uploaded",
                    errors=[{"field": "file", "message": "File required"}]
                )
            file_name = f"{uuid.uuid4().hex}_{file.filename}"
            file_path = kb_path / file_name
            with open(file_path, "wb") as f_out:
                f_out.write(await file.read())
            source_details = {
                "type": "document",
                "source_id": kb.workspace_id,
                "filename": file_name,
                "file_url": str(file_path)
            }

        elif source_type == "web_page":
            if not url:
                return format_response(
                    status=False,
                    message="No URL provided",
                    errors=[{"field": "url", "message": "URL is required"}]
                )
            file_name = f"web_{uuid.uuid4().hex}.txt"
            file_path = kb_path / file_name
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"Crawled content from {url}")
            source_details = {
                "type": "web_page",
                "source_id": kb.workspace_id,
                "filename": file_name,
                "file_url": str(file_path)
            }

        elif source_type == "text":
            if not text_filename or not text_content:
                return format_response(
                    status=False,
                    message="Filename and content required",
                    errors=[{"field": "text", "message": "Both filename and content are required"}]
                )
            file_name = f"{uuid.uuid4().hex}_{text_filename}.txt"
            file_path = kb_path / file_name
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text_content)
            source_details = {
                "type": "text",
                "source_id": kb.workspace_id,
                "filename": file_name,
                "file_url": str(file_path)
            }

        else:
            return format_response(
                status=False,
                message="Invalid source type",
                errors=[{"field": "source_type", "message": "Must be one of: file, web_page, text"}],
                status_code=400
            )

        # Optionally store source in KnowledgeFile
        if source_type == "file":
            kb_file = KnowledgeFile(
                kb_id=kb.id,
                filename=file_name,
                file_path=str(file_path)
            )
            db.add(kb_file)
            db.commit()
            db.refresh(kb_file)

        return format_response(
            status=True,
            message="Knowledge base source added",
            data=source_details
        )

    except Exception as e:
        db.rollback()
        return format_response(
            status=False,
            message="Internal Server Error",
            errors=[{"field": "server", "message": str(e)}]
        )

@app.delete("/knowledge-bases/{kb_id}")
def delete_kb(
    kb_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        kb = db.query(KnowledgeBase).filter_by(id=kb_id).first()
        if not kb:
            return format_response(
                status=False,
                message="Knowledge base not found",
                errors=[{"field": "kb_id", "message": "Knowledge base not found"}],
                status_code=404
            )

        workspace = db.query(Workspace).filter_by(id=kb.workspace_id).first()
        if workspace.owner_id != current_user.id:
            print(f"DEBUG: current_user.id = {current_user.id}, workspace.owner_id = {workspace.owner_id}")
            return format_response(
                status=False,
                message="Only owner can delete",
                errors=[{"field": "permission", "message": "Only the workspace owner can delete this knowledge base"}],
                status_code=403
            )

        db.query(KnowledgeFile).filter_by(kb_id=kb_id).delete()
        db.delete(kb)
        db.commit()

        kb_path = UPLOAD_DIR / str(workspace.id) / str(kb.id)
        if kb_path.exists():
            import shutil
            shutil.rmtree(kb_path)

        return format_response(
            status=True,
            message="Knowledge base deleted"
        )

    except Exception as e:
        db.rollback()
        return format_response(
            status=False,
            message="Internal Server Error",
            errors=[{"field": "server", "message": str(e)}]
        )

@app.delete("/knowledge-files/{file_id}")
def delete_kb_file(
    file_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        file = db.query(KnowledgeFile).filter_by(id=file_id).first()
        if not file:
            return format_response(
                status=False,
                message="File not found",
                errors=[{"field": "file_id", "message": "Knowledge file not found"}],
                status_code=404
            )

        kb = db.query(KnowledgeBase).filter_by(id=file.kb_id).first()
        membership = db.query(WorkspaceMember).filter_by(workspace_id=kb.workspace_id, user_id=current_user.id).first()
        if not membership:
            return format_response(
                status=False,
                message="Access denied",
                errors=[{"field": "workspace_id", "message": "User is not a member of this workspace"}],
                status_code=403
            )

        try:
            os.remove(file.file_path)
        except:
            pass

        db.delete(file)
        db.commit()

        return format_response(
            status=True,
            message="File deleted"
        )

    except Exception as e:
        db.rollback()
        return format_response(
            status=False,
            message="Internal Server Error",
            errors=[{"field": "server", "message": str(e)}]
        )
    
@app.get("/workspaces/{workspace_id}/settings")
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

@app.put("/workspaces/{workspace_id}/settings")
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
        
@app.get("/list-knowledge-bases")
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
@app.post("/workspaces/{workspace_id}/api-keys")
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

@app.get("/workspaces/{workspace_id}/api-keys")
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

@app.delete("/workspaces/{workspace_id}/api-keys/{key_id}")
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

@app.put("/api-keys/{key_id}/rename")
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

@app.put("/api-keys/{key_id}/set-webhook")
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


