from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from app.schemas import agent as schemas_agent
from app.services import agent_service
from app.core.dependencies import get_db, get_current_company, get_current_active_user, require_permission
from app.models import user as models_user
from app.crud import crud_published_widget_settings
from app.services import widget_settings_service
from app.schemas import widget_settings as schemas_widget_settings


router = APIRouter()

@router.get("/{agent_id}/widget-settings", response_model=schemas_widget_settings.WidgetSettings)
def read_widget_settings(agent_id: int, db: Session = Depends(get_db)):
    return widget_settings_service.get_widget_settings(db, agent_id=agent_id)

@router.put("/{agent_id}/widget-settings", response_model=schemas_widget_settings.WidgetSettings)
def update_widget_settings(agent_id: int, widget_settings: schemas_widget_settings.WidgetSettingsUpdate, db: Session = Depends(get_db)):
    return widget_settings_service.update_widget_settings(db, agent_id=agent_id, widget_settings=widget_settings)

@router.post("/{agent_id}/publish", status_code=status.HTTP_201_CREATED, dependencies=[Depends(require_permission("agent:update"))])
def publish_agent_settings(
    agent_id: int,
    settings: dict,
    db: Session = Depends(get_db),
    current_user: models_user.User = Depends(get_current_active_user)
):
    # Optional: Add validation to ensure the agent belongs to the user's company
    db_agent = agent_service.get_agent(db, agent_id=agent_id, company_id=current_user.company_id)
    if not db_agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    published_settings = crud_published_widget_settings.create_published_widget_settings(db, settings=settings)
    return {"publish_id": published_settings.publish_id}

@router.post("/", response_model=schemas_agent.Agent, dependencies=[Depends(require_permission("agent:create"))])
def create_agent(agent: schemas_agent.AgentCreate, db: Session = Depends(get_db), current_user: models_user.User = Depends(get_current_active_user)):
    return agent_service.create_agent(db=db, agent=agent, company_id=current_user.company_id)

@router.get("", response_model=List[schemas_agent.Agent], dependencies=[Depends(require_permission("agent:read"))])
def read_agents(skip: int = 0, limit: int = 100, db: Session = Depends(get_db), current_user: models_user.User = Depends(get_current_active_user)):
    # Only return active agents by default, or all if a query parameter is set
    agents = agent_service.get_agents(db, company_id=current_user.company_id, skip=skip, limit=limit)
    return agents

router.add_api_route(
    "/",
    read_agents,
    response_model=List[schemas_agent.Agent],
    dependencies=[Depends(require_permission("agent:read"))],
    methods=["GET"],
    include_in_schema=False  # To avoid duplicate entries in the OpenAPI schema
)

@router.get("/{agent_id}", response_model=schemas_agent.Agent, dependencies=[Depends(require_permission("agent:read"))])
def read_agent(agent_id: int, db: Session = Depends(get_db), current_user: models_user.User = Depends(get_current_active_user)):
    db_agent = agent_service.get_agent(db, agent_id=agent_id, company_id=current_user.company_id)
    if db_agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    return db_agent

@router.put("/{agent_id}", response_model=schemas_agent.Agent, dependencies=[Depends(require_permission("agent:update"))])
def update_agent(agent_id: int, agent: schemas_agent.AgentUpdate, db: Session = Depends(get_db), current_user: models_user.User = Depends(get_current_active_user)):
    db_agent = agent_service.update_agent(db, agent_id=agent_id, agent=agent, company_id=current_user.company_id)
    if db_agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    return db_agent

@router.delete("/{agent_id}", dependencies=[Depends(require_permission("agent:delete"))])
def delete_agent(agent_id: int, db: Session = Depends(get_db), current_user: models_user.User = Depends(get_current_active_user)):
    db_agent = agent_service.delete_agent(db, agent_id=agent_id, company_id=current_user.company_id)
    if db_agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"message": "Agent deleted successfully"}

@router.post("/{agent_id}/new-version", response_model=schemas_agent.Agent, status_code=status.HTTP_201_CREATED, dependencies=[Depends(require_permission("agent:update"))])
def create_agent_new_version(
    agent_id: int,
    db: Session = Depends(get_db),

    current_user: models_user.User = Depends(get_current_active_user)
):
    try:
        new_version = agent_service.create_agent_version(db, agent_id, current_user.company_id)
        return new_version
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.put("/{agent_id}/activate-version", response_model=schemas_agent.Agent, dependencies=[Depends(require_permission("agent:update"))])
def activate_agent_version(
    agent_id: int,
    db: Session = Depends(get_db),

    current_user: models_user.User = Depends(get_current_active_user)
):
    try:
        activated_agent = agent_service.activate_agent_version(db, agent_id, current_user.company_id)
        return activated_agent
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
