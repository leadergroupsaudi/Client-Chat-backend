
from sqlalchemy.orm import Session
from app.models.widget_settings import WidgetSettings
from app.models.agent import Agent
from app.schemas.widget_settings import WidgetSettingsCreate, WidgetSettingsUpdate
from app.core.config import settings

def get_widget_settings(db: Session, agent_id: int):
    settings_from_db = db.query(WidgetSettings).filter(WidgetSettings.agent_id == agent_id).first()
    if settings_from_db:
        if not settings_from_db.livekit_url:
            settings_from_db.livekit_url = settings.LIVEKIT_URL
        if not settings_from_db.frontend_url:
            settings_from_db.frontend_url = settings.FRONTEND_URL

        # Fetch agent to include voice settings (voice_id, stt_provider, tts_provider)
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if agent:
            # Add agent's voice settings to the widget settings object
            # These are transient attributes - not persisted to widget_settings table
            settings_from_db.voice_id = agent.voice_id
            settings_from_db.stt_provider = agent.stt_provider
            settings_from_db.tts_provider = agent.tts_provider

        return settings_from_db
    return None

def create_widget_settings(db: Session, widget_settings: WidgetSettingsCreate):
    widget_settings_data = widget_settings.model_dump()
    widget_settings_data["livekit_url"] = settings.LIVEKIT_URL
    widget_settings_data["frontend_url"] = settings.FRONTEND_URL
    
    # These fields are not part of the WidgetSettings model, so remove them
    widget_settings_data.pop("voice_id", None)
    widget_settings_data.pop("stt_provider", None)
    widget_settings_data.pop("tts_provider", None)
    
    db_widget_settings = WidgetSettings(**widget_settings_data)
    db.add(db_widget_settings)
    db.commit()
    db.refresh(db_widget_settings)
    return db_widget_settings

def update_widget_settings(db: Session, agent_id: int, widget_settings: WidgetSettingsUpdate):
    db_widget_settings = get_widget_settings(db, agent_id)
    if db_widget_settings:
        update_data = widget_settings.model_dump(exclude_unset=True)

        # These fields are not part of the WidgetSettings model, so remove them
        update_data.pop("voice_id", None)
        update_data.pop("stt_provider", None)
        update_data.pop("tts_provider", None)

        for key, value in update_data.items():
            setattr(db_widget_settings, key, value)
        db.commit()
        db.refresh(db_widget_settings)
    return db_widget_settings
