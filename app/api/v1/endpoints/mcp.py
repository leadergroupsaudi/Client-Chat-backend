import json
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import Dict, Any, Optional
from fastmcp.client import Client
from sqlalchemy.orm import Session

from app.core.dependencies import get_db, get_current_user
from app.models.user import User
from app.services import credential_service
from app.services.vault_service import vault_service

router = APIRouter()


class McpInspectRequest(BaseModel):
    url: HttpUrl


class McpExecuteRequest(BaseModel):
    url: HttpUrl
    tool_name: str
    parameters: Dict[str, Any]


def get_access_token_if_available(db: Session, company_id: int) -> Optional[str]:
    """Return Google access token string if available, else None."""
    google_credential = credential_service.get_credential_by_service_name(
        db, service_name="google", company_id=company_id
    )
    if google_credential:
        try:
            decrypted_creds = vault_service.decrypt(google_credential.encrypted_credentials)
            access_token = json.loads(decrypted_creds).get("token")
            if access_token:
                return access_token
        except Exception as e:
            print(f"Could not decrypt credentials, proceeding without them. Error: {e}")
    return None


def _build_auth_kwargs(token: Optional[str]) -> dict:
    """Build auth kwargs for FastMCP Client — only inject if token is present."""
    if token:
        return {"auth": f"Bearer {token}"}
    return {}


def _extract_result_text(result) -> str:
    """Extract plain text from a FastMCP call_tool result (list of content items).

    FastMCP wraps tool return values in TextContent objects whose `.text` field
    holds either the raw string or a JSON-serialised representation.
    The previous fallback used str(item) which printed Pydantic model reprs;
    now we JSON-serialise unknown items so the output is always readable.
    """
    if result is None:
        return ""
    import json as _json
    parts = []
    items = result if isinstance(result, list) else [result]
    for item in items:
        if hasattr(item, "text"):
            parts.append(item.text)
        elif hasattr(item, "content"):
            # EmbeddedResource — recurse one level
            inner = item.content
            parts.append(inner.text if hasattr(inner, "text") else _json.dumps(inner, default=str))
        else:
            try:
                parts.append(_json.dumps(item, default=str))
            except Exception:
                parts.append(str(item))
    return "\n".join(parts)


@router.post("/inspect")
async def inspect_mcp_server(
    request: McpInspectRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    token = get_access_token_if_available(db, current_user.company_id)
    auth_kwargs = _build_auth_kwargs(token)

    try:
        async with Client(str(request.url), **auth_kwargs) as client:
            tool_list = await client.list_tools()

        tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema,
            }
            for tool in tool_list
        ]
        return {"tools": tools}

    except Exception as e:
        url_str = str(request.url)
        # Only suggest Google auth when the target IS a Google service and no
        # token is stored — not for generic connection failures to local servers.
        is_google_url = "googleapis.com" in url_str or "google.com" in url_str
        if is_google_url and token is None:
            return {
                "authentication_required": True,
                "authorization_url": "https://accounts.google.com/o/oauth2/v2/auth",
                "message": "This tool requires Google authentication. Please connect your Google account to proceed.",
            }
        raise HTTPException(
            status_code=400,
            detail=f"Failed to connect or inspect MCP server. Error: {str(e)}",
        )


@router.post("/execute")
async def execute_mcp_tool(
    request: McpExecuteRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    token = get_access_token_if_available(db, current_user.company_id)
    auth_kwargs = _build_auth_kwargs(token)

    try:
        async with Client(str(request.url), **auth_kwargs) as client:
            result = await client.call_tool(request.tool_name, request.parameters)
        return {"result": _extract_result_text(result)}
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to execute tool on MCP server. Error: {str(e)}",
        )
