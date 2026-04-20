"""
SQLAlchemy model for dynamic voice agent configurations.
Stores user-defined prompts and tool selections that drive dynamic behavior.
"""
from sqlalchemy import Column, Integer, String, Text, Boolean, JSON, DateTime
from datetime import datetime
from app.core.database import Base


class VoiceAgentConfig(Base):
    __tablename__ = "voice_agent_configs"

    id = Column(Integer, primary_key=True, index=True)
    company_id = Column(Integer, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    system_prompt = Column(Text, nullable=True)
    user_prompt = Column(Text, nullable=False)
    selected_tools = Column(JSON, default=list, nullable=False)
    voice_enabled = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )
