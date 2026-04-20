"""
Pydantic schemas for VoiceAgentConfig.
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class VoiceAgentConfigBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    system_prompt: Optional[str] = Field(None, description="Optional base context/constraints")
    user_prompt: str = Field(..., min_length=1, description="Mandatory behavior instructions for the agent")
    selected_tools: List[str] = Field(default_factory=list, description="List of tool names to make available")
    voice_enabled: bool = Field(default=True)


class VoiceAgentConfigCreate(VoiceAgentConfigBase):
    pass


class VoiceAgentConfigUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = Field(None, min_length=1)
    selected_tools: Optional[List[str]] = None
    voice_enabled: Optional[bool] = None


class VoiceAgentConfigRead(VoiceAgentConfigBase):
    id: int
    company_id: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class AvailableTool(BaseModel):
    name: str
    label: str
    description: str


class SessionTokenResponse(BaseModel):
    token: str
    model: str
    instructions: str
    tools: List[dict]


class ExecuteToolRequest(BaseModel):
    tool_name: str
    tool_args: dict = Field(default_factory=dict)


class ExecuteToolResponse(BaseModel):
    status: str
    result: dict
