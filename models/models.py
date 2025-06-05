from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, func
from sqlalchemy.orm import relationship
from datetime import datetime
from db.database import Base

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
