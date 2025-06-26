from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float, JSON, BigInteger, ForeignKey, func
from sqlalchemy.orm import relationship
from datetime import datetime
from db.database import Base
from sqlalchemy.dialects.postgresql import JSONB
import uuid, enum, time
from sqlalchemy import Enum as SqlEnum
from sqlalchemy.dialects.postgresql import ARRAY

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

class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"

    id = Column(String, primary_key=True, index=True, default=lambda: f"{uuid.uuid4().hex[:16]}")
    name = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    workspace_id = Column(Integer, ForeignKey("workspaces.id"), nullable=False)
    enable_auto_refresh = Column(Boolean, default=False, nullable=False)
    auto_refresh_interval = Column(Integer, default=24, nullable=False)  # hours
    last_refreshed = Column(DateTime, nullable=True)

    workspace = relationship("Workspace", back_populates="knowledge_bases")

class FileStatus(str, enum.Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"

class SourceStatus(str, enum.Enum):
    file = "file"
    url = "url"
    txt = "txt"

class KnowledgeFile(Base):
    __tablename__ = "knowledge_files"
    id = Column(Integer, primary_key=True, index=True)
    kb_id = Column(String, ForeignKey("knowledge_bases.id"), nullable=False) 
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=True)
    extract_data = Column(Text, nullable=True)
    status = Column(SqlEnum(FileStatus), default=FileStatus.pending, nullable=False)
    source_type = Column(SqlEnum(SourceStatus), default=SourceStatus.file, nullable=False)
    embedding = Column(ARRAY(Float), nullable=True)
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

# Agent Models
class pbx_ai_agent(Base):
    __tablename__ = "pbx_ai_agent"

    id = Column(String, primary_key=True, index=True, default=lambda: f"{uuid.uuid4().hex[:16]}")
    version = Column(Integer, default=0)
    workspace_id = Column(Integer, ForeignKey("workspaces.id"), nullable=False)
    name = Column(String, nullable=True)
    voice_id = Column(String)
    voice_model = Column(String, nullable=True)
    fallback_voice_ids = Column(JSON, nullable=True)
    voice_temperature = Column(Float, default=1.0)
    voice_speed = Column(Float, default=1.0)
    volume = Column(Float, default=1.0)
    responsiveness = Column(Float, default=1.0)
    interruption_sensitivity = Column(Float, default=1.0)
    enable_backchannel = Column(Boolean, default=False)
    backchannel_frequency = Column(Float, default=0.8)
    backchannel_words = Column(JSON, nullable=True)
    reminder_trigger_ms = Column(Integer, default=10000)
    reminder_max_count = Column(Integer, default=1)
    ambient_sound = Column(String, nullable=True)
    ambient_sound_volume = Column(Float, default=1.0)
    language = Column(String, default="en-US")
    webhook_url = Column(String, nullable=True)
    boosted_keywords = Column(JSON, nullable=True)
    opt_out_sensitive_data_storage = Column(Boolean, default=False)
    opt_in_signed_url = Column(Boolean, default=True)
    pronunciation_dictionary = Column(JSON, nullable=True)
    normalize_for_speech = Column(Boolean, default=True)
    end_call_after_silence_ms = Column(Integer, default=600000)
    max_call_duration_ms = Column(Integer, default=3600000)
    voicemail_option = Column(JSON, nullable=True)
    post_call_analysis_data = Column(JSON, nullable=True)
    post_call_analysis_model = Column(String, default="gpt-4o-mini")
    begin_message_delay_ms = Column(Integer, default=1000)
    ring_duration_ms = Column(Integer, default=30000)
    stt_mode = Column(String, default="fast")
    vocab_specialization = Column(String, default="general")
    allow_user_dtmf = Column(Boolean, default=True)
    user_dtmf_options = Column(JSON, nullable=True)
    denoising_mode = Column(String, default="noise-cancellation")
    last_modification_timestamp = Column(BigInteger, default=lambda: int(time.time() * 1000))
    is_published = Column(Boolean, default=False)
    response_engine = Column(JSON)

# PBX LLm
class PBXLLM(Base):
    __tablename__ = "pbx_llms"

    id = Column(String, primary_key=True, default=lambda: f"{uuid.uuid4().hex[:24]}")
    workspace_id = Column(Integer, ForeignKey("workspaces.id"), nullable=False)

    version = Column(Integer, default=0)
    model = Column(String, nullable=True)
    s2s_model = Column(String, nullable=True)
    model_temperature = Column(Float, default=0.0)
    model_high_priority = Column(Boolean, default=False)
    tool_call_strict_mode = Column(Boolean, default=False)

    general_prompt = Column(Text, nullable=True)  # Prefer `Text` for long prompts
    general_tools = Column(JSON, nullable=True)
    states = Column(JSON, nullable=True)
    starting_state = Column(String, nullable=True)
    begin_message = Column(String, nullable=True)
    default_dynamic_variables = Column(JSON, nullable=True)
    knowledge_base_ids = Column(JSON, nullable=True)

    is_published = Column(Boolean, default=False)
    last_modification_timestamp = Column(BigInteger, default=lambda: int(time.time() * 1000))

# Chat-room
class ChatSession(Base):
    __tablename__ = "chat_sessions"

    chat_id = Column(String, primary_key=True, default=lambda: f"{uuid.uuid4().hex[:24]}")
    agent_id = Column(String, nullable=False)
    agent_version = Column(Integer, default=0)
    chat_status = Column(String, default="ongoing")  # ongoing, ended, error
    retell_llm_dynamic_variables = Column(JSONB, nullable=True)
    collected_dynamic_variables = Column(JSONB, nullable=True)
    start_timestamp = Column(BigInteger, nullable=True)
    end_timestamp = Column(BigInteger, nullable=True)
    transcript = Column(String, nullable=True)
    message_with_tool_calls = Column(JSONB, nullable=True)
    chat_metadata = Column("metadata", JSONB, nullable=True)  # renamed to avoid SQLAlchemy conflict
    chat_cost = Column(JSONB, nullable=True)
    chat_analysis = Column(JSONB, nullable=True)

# voice
class LLMVoice(Base):
    __tablename__ = "voices"

    voice_id = Column(String, primary_key=True, index=True)
    voice_name = Column(String, nullable=False)
    provider = Column(String, nullable=False)
    gender = Column(String, nullable=False)
    accent = Column(String, nullable=True)
    age = Column(String, nullable=True)
    preview_audio_url = Column(String, nullable=True)

#import phone
class ImportedPhoneNumber(Base):
    __tablename__ = "imported_phone_numbers"

    id = Column(Integer, primary_key=True, index=True)
    phone_number = Column(String, unique=True, nullable=False, index=True)
    phone_number_type = Column(String, default="retell-twilio")
    phone_number_pretty = Column(String)
    inbound_agent_id = Column(String, nullable=True)
    outbound_agent_id = Column(String, nullable=True)
    inbound_agent_version = Column(Integer, nullable=True)
    outbound_agent_version = Column(Integer, nullable=True)
    area_code = Column(Integer)
    nickname = Column(String, nullable=True)
    inbound_webhook_url = Column(String, nullable=True)
    last_modification_timestamp = Column(BigInteger)