from sqlalchemy import Column, Integer, String, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

Base = declarative_base()


class SIPTrunk(Base):
    __tablename__ = 'sip_trunks'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    numbers = Column(JSON, nullable=False)
    auth_username = Column(String, nullable=False)
    auth_password = Column(String, nullable=False)
    livekit_trunk_id = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class SIPDispatchRule(Base):
    __tablename__ = 'sip_dispatch_rules'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    room_prefix = Column(String, nullable=False)
    agent_name = Column(String, nullable=False)
    meta_data = Column(String, nullable=True)
    livekit_dispatch_id = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class CallSession(Base):
    __tablename__ = 'call_sessions'

    id = Column(Integer, primary_key=True, index=True)
    room_id = Column(String, nullable=False, unique=True)
    caller_number = Column(String, nullable=False)
    agent_identity = Column(String, nullable=True)
    status = Column(String, default='initiated')  # initiated, connected, ended
    created_at = Column(DateTime(timezone=True), server_default=func.now())
