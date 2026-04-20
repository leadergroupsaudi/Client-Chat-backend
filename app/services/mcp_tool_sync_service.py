"""
MCP Tool Sync Service

Runs at backend startup. Connects to the local Automax MCP server,
discovers its tools via list_tools, and creates/updates a single Tool
DB record so agents can be assigned the MCP connection.
"""
from sqlalchemy.orm import Session
from app.models.tool import Tool

MCP_SERVER_URL = "http://localhost:8002/mcp"
MCP_TOOL_NAME = "automax_mcp"


async def sync_mcp_tools(db: Session) -> None:
    """
    Async — must be awaited from an async context (e.g. FastAPI startup).

    Bug fixed: the original sync wrapper called asyncio.run() inside FastAPI's
    async on_startup handler, which already has a running event loop.
    asyncio.run() raises RuntimeError when an event loop is already running.
    The function is now async so callers simply `await` it.
    """
    from fastmcp.client import Client

    try:
        async with Client(MCP_SERVER_URL) as client:
            tool_list = await client.list_tools()

        tool_names = [t.name for t in tool_list]
        description = (
            f"Automax Incident Management MCP Server — "
            f"{len(tool_names)} tools: {', '.join(tool_names)}"
        )

        # Bug fixed: also filter by company_id IS NULL so a company-specific
        # tool named "automax_mcp" doesn't shadow the global record.
        existing = (
            db.query(Tool)
            .filter(Tool.name == MCP_TOOL_NAME, Tool.company_id.is_(None))
            .first()
        )

        if existing:
            existing.mcp_server_url = MCP_SERVER_URL
            existing.description = description
            existing.tool_type = "mcp"
        else:
            db_tool = Tool(
                name=MCP_TOOL_NAME,
                description=description,
                tool_type="mcp",
                mcp_server_url=MCP_SERVER_URL,
                company_id=None,  # Global — visible to all companies
            )
            db.add(db_tool)

        db.commit()
        print(
            f"[MCP Sync] ✅ Synced {len(tool_names)} tools from {MCP_SERVER_URL}: "
            f"{', '.join(tool_names)}"
        )

    except Exception as e:
        # Non-fatal — MCP server may not be running yet
        print(f"[MCP Sync] ⚠️  Could not sync MCP tools (server may be offline): {e}")
