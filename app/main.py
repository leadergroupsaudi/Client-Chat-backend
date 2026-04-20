from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.core.database import Base, engine, SessionLocal
from app.models import role, permission, contact, comment # Import new models
from app.core.config import settings
from app.api.v1.main import api_router, websocket_router
from app.api.v1.endpoints import ws_updates, comments, gmail, google, published, ai_images, ai_chat, object_detection, public_pages
from app.core.dependencies import get_db
from app.services.connection_manager import manager
from app.services.websocket_cleanup_service import cleanup_inactive_sessions
from app.services.call_timeout_service import call_timeout_service
from app.services import tool_service, widget_settings_service
from app.schemas import widget_settings as schemas_widget_settings
from create_tool import create_api_call_tool
import asyncio

# Create all database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
)

# CORS middleware to allow frontend to connect
# Note: Widget needs to be accessible from any origin
# Parse CORS origins from comma-separated string in settings
cors_origins = [origin.strip() for origin in settings.CORS_ORIGINS.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.include_router(api_router, prefix=settings.API_V1_STR)
app.include_router(ws_updates.router, prefix="/ws") # New company-wide updates
app.include_router(comments.router, prefix="/api/v1/comments")
app.include_router(gmail.router, prefix="/api/v1/gmail")
app.include_router(google.router, prefix="/api/v1/google")
app.include_router(published.router, prefix="/api/v1/published")
app.include_router(ai_images.router, prefix="/api/v1/ai-images", tags=["ai-images"])
app.include_router(ai_chat.router, prefix="/api/v1/ai-chat", tags=["ai-chat"])
app.include_router(object_detection.router, prefix="/api/v1/object-detection", tags=["object-detection"])
app.include_router(public_pages.router, tags=["public-pages"])  # Public pages (privacy policy, terms) at root

# Mount static files for serving the widget
widget_static_path = os.path.join(os.path.dirname(__file__), "static", "widget")
if os.path.exists(widget_static_path):
    app.mount("/widget", StaticFiles(directory=widget_static_path), name="widget")

@app.get("/")
async def read_root():
    return {"message": "AgentConnect backend is running"}


from app.initial_data import create_initial_data
from app.services.builtin_tools_service import seed_builtin_tools
from app.services.mcp_tool_sync_service import sync_mcp_tools

# Initialize scheduler for background tasks
scheduler = AsyncIOScheduler()

async def run_campaign_scheduler():
    """Wrapper to run the campaign scheduler with a fresh DB session"""
    from app.services import campaign_execution_service
    db = SessionLocal()
    try:
        await campaign_execution_service.process_all_scheduled_campaigns(db)
    finally:
        db.close()


async def run_whatsapp_token_refresh():
    """Wrapper to run the WhatsApp token refresh with a fresh DB session"""
    from app.services.whatsapp_token_refresh_service import run_whatsapp_token_refresh_scheduler
    await run_whatsapp_token_refresh_scheduler()


@app.on_event("startup")
async def on_startup():
    create_initial_data()

    # Seed built-in tools (idempotent)
    db = SessionLocal()
    try:
        seed_builtin_tools(db)
    finally:
        db.close()

    # Sync MCP tools from Automax MCP server (non-fatal if server is offline)
    db = SessionLocal()
    try:
        await sync_mcp_tools(db)
    finally:
        db.close()

    # Start WebSocket cleanup scheduler if enabled
    if settings.WS_ENABLE_HEARTBEAT:
        scheduler.add_job(
            cleanup_inactive_sessions,
            'interval',
            seconds=settings.WS_CLEANUP_INTERVAL,
            args=[manager],
            id='websocket_cleanup',
            replace_existing=True
        )
        print(f"[Startup] WebSocket cleanup scheduler started (interval: {settings.WS_CLEANUP_INTERVAL}s)")
        print(f"[Startup] Preview session timeout: {settings.WS_PREVIEW_SESSION_TIMEOUT}s")
        print(f"[Startup] Regular session timeout: {settings.WS_REGULAR_SESSION_TIMEOUT}s")
    else:
        print("[Startup] WebSocket heartbeat disabled (WS_ENABLE_HEARTBEAT=False)")

    # Add campaign scheduler job - runs every 30 seconds to process scheduled campaigns
    scheduler.add_job(
        run_campaign_scheduler,
        'interval',
        seconds=30,
        id='campaign_scheduler',
        replace_existing=True
    )
    print("[Startup] Campaign scheduler started (interval: 30s)")

    # Add WhatsApp token refresh job - runs every hour to refresh expiring tokens
    scheduler.add_job(
        run_whatsapp_token_refresh,
        'interval',
        hours=1,
        id='whatsapp_token_refresh',
        replace_existing=True
    )
    print("[Startup] WhatsApp token refresh scheduler started (interval: 1 hour)")

    # Start the scheduler if not already started
    if not scheduler.running:
        scheduler.start()

    # Start call timeout service
    asyncio.create_task(call_timeout_service.start())
    print("[Startup] Call timeout service started (timeout: 30s, check interval: 10s)")

@app.on_event("shutdown")
async def on_shutdown():
    print("Server is shutting down...")

    # Stop call timeout service
    call_timeout_service.stop()
    print("[Shutdown] Call timeout service stopped")

    # Shutdown scheduler
    if scheduler.running:
        scheduler.shutdown(wait=False)
        print("[Shutdown] Scheduler stopped")

    # Disconnect all WebSocket clients
    await manager.disconnect_all()
    print("[Shutdown] All WebSocket clients disconnected")

if __name__ == "__main__":
    uvicorn.run(app, host=settings.HOST, port=settings.PORT, ws="websockets")