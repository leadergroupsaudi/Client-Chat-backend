from typing import Optional
from pydantic import BaseModel, Field

class ContactInfo(BaseModel):
    """Contact information embedded in session response"""
    id: Optional[int] = None
    name: Optional[str] = None
    email: Optional[str] = None
    phone_number: Optional[str] = None

    class Config:
        from_attributes = True

class Session(BaseModel):
    session_id: str = Field(..., alias="conversation_id")
    status: str
    assignee_id: Optional[int] = None
    last_message_timestamp: str
    first_message_content: str
    channel: Optional[str] = None
    contact_name: Optional[str] = None
    contact_phone: Optional[str] = None
    contact: Optional[ContactInfo] = None  # Full contact object
    is_client_connected: Optional[bool] = False
    is_ai_enabled: Optional[bool] = True
    priority: Optional[int] = 0  # 0=None, 1=Low, 2=Medium, 3=High, 4=Urgent
    workflow_id: Optional[int] = None

    class Config:
        populate_by_name = True
        from_attributes = True
