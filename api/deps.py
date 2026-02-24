from __future__ import annotations

from fastapi import Depends, Header, HTTPException

from core.config import settings


def _api_key_dep(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    expected = (settings.api_key or "").strip()
    if not expected:
        return
    if (x_api_key or "").strip() != expected:
        raise HTTPException(status_code=401, detail="invalid_api_key")


ApiKeyDep = Depends(_api_key_dep)
