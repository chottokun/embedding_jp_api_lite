"""Optional Bearer API key authentication."""

import secrets
from typing import Optional

from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import get_api_key

security = HTTPBearer(auto_error=False)


async def verify_api_key(
    auth: Optional[HTTPAuthorizationCredentials] = Security(security),
) -> Optional[HTTPAuthorizationCredentials]:
    """Verify Bearer token when API_KEY is configured; allow all requests otherwise."""
    configured_key = get_api_key()
    if not configured_key:
        return auth

    if auth is None or not secrets.compare_digest(auth.credentials, configured_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API Key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return auth
