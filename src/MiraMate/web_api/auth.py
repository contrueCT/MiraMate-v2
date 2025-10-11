"""
极简鉴权模块：
- 当且仅当设置了环境变量 MIRAMATE_AUTH_TOKEN 时启用鉴权；
- HTTP：要求 Authorization: Bearer <token>；
- WebSocket：要求连接 URL query 参数携带 token=<token>；
- 白名单仅放行 /api/health 以减少服务信息暴露。
"""

from __future__ import annotations

import os
from typing import Set
from fastapi import Request, HTTPException, status

TOKEN_ENV = "MIRAMATE_AUTH_TOKEN"

# 仅放行健康检查
WHITELIST_PATHS: Set[str] = {
    "/api/health",
}


def is_auth_enabled() -> bool:
    token = os.getenv(TOKEN_ENV, "").strip()
    return bool(token)


def get_expected_token() -> str:
    return os.getenv(TOKEN_ENV, "").strip()


def should_skip_auth(path: str) -> bool:
    return path in WHITELIST_PATHS


def verify_http_request(request: Request) -> None:
    """校验 HTTP 请求的 Bearer Token，不通过则抛出 401。"""
    if not is_auth_enabled():
        return
    if should_skip_auth(request.url.path):
        return

    auth_header = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth_header or not auth_header.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Bearer"},
        )
    provided = auth_header.split(" ", 1)[1].strip()
    expected = get_expected_token()
    if provided != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Bearer"},
        )


def verify_token_value(token: str | None) -> bool:
    """用于 WS 的 token 比对。"""
    if not is_auth_enabled():
        return True
    expected = get_expected_token()
    return bool(token) and token == expected
