"""
CRUD operations for VoiceAgentConfig.
"""
from sqlalchemy.orm import Session
from typing import List, Optional

from app.models.voice_agent_config import VoiceAgentConfig
from app.schemas.voice_agent_config import VoiceAgentConfigCreate, VoiceAgentConfigUpdate


def get_all(db: Session, company_id: int) -> List[VoiceAgentConfig]:
    return (
        db.query(VoiceAgentConfig)
        .filter(VoiceAgentConfig.company_id == company_id)
        .order_by(VoiceAgentConfig.created_at.desc())
        .all()
    )


def get_by_id(db: Session, config_id: int, company_id: int) -> Optional[VoiceAgentConfig]:
    return (
        db.query(VoiceAgentConfig)
        .filter(
            VoiceAgentConfig.id == config_id,
            VoiceAgentConfig.company_id == company_id,
        )
        .first()
    )


def create(db: Session, data: VoiceAgentConfigCreate, company_id: int) -> VoiceAgentConfig:
    config = VoiceAgentConfig(**data.model_dump(), company_id=company_id)
    db.add(config)
    db.commit()
    db.refresh(config)
    return config


def update(db: Session, config: VoiceAgentConfig, data: VoiceAgentConfigUpdate) -> VoiceAgentConfig:
    updates = data.model_dump(exclude_unset=True)
    for field, value in updates.items():
        setattr(config, field, value)
    db.commit()
    db.refresh(config)
    return config


def delete(db: Session, config: VoiceAgentConfig) -> None:
    db.delete(config)
    db.commit()
