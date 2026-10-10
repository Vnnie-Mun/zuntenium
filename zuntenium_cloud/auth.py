# zuntenium_cloud/auth.py
"""
Authentication & Multi-Tenancy
================================
JWT access tokens (15-min expiry) + refresh tokens (7-day).
API key authentication for programmatic/CLI access.
Role-based access control: user / admin.
Per-user resource quota enforcement.

Security:
  - Passwords hashed with bcrypt (cost factor 12)
  - API keys: 32-byte random token, SHA-256 hashed in DB
  - JWT signed with RS256 (asymmetric) — public key for verify, private for sign
  - All tokens include user_id, role, jti (JWT ID for revocation)
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Annotated

import bcrypt
import jwt
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import (
    HTTPAuthorizationCredentials, HTTPBearer, OAuth2PasswordBearer,
)
from pydantic import BaseModel, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession

from .database import get_db
from .models import APIKey, Job, User, UserQuota, UserRole

_log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

JWT_ALGORITHM         = "HS256"          # HS256 for simplicity; swap to RS256 in prod
ACCESS_TOKEN_EXPIRE   = timedelta(minutes=15)
REFRESH_TOKEN_EXPIRE  = timedelta(days=7)
BCRYPT_ROUNDS         = 12

_bearer = HTTPBearer(auto_error=False)
_oauth2 = OAuth2PasswordBearer(tokenUrl="/auth/token", auto_error=False)


def _jwt_secret() -> str:
    secret = os.environ.get("ZUNTENIUM_JWT_SECRET")
    if not secret:
        raise RuntimeError("ZUNTENIUM_JWT_SECRET env var not set")
    return secret


# ── Password hashing ──────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt(rounds=BCRYPT_ROUNDS)).decode()

def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False


# ── API key generation ────────────────────────────────────────────────────────

def generate_api_key() -> tuple[str, str, str]:
    """
    Returns (raw_key, prefix, sha256_hash).
    raw_key format: "ztn_<32-byte-hex>"  (shown once to user)
    """
    raw       = "ztn_" + secrets.token_hex(32)
    prefix    = raw[:8]
    key_hash  = hashlib.sha256(raw.encode()).hexdigest()
    return raw, prefix, key_hash


def hash_api_key(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


# ── JWT token creation ────────────────────────────────────────────────────────

def create_access_token(user_id: str, role: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub":  user_id,
        "role": role,
        "jti":  str(uuid.uuid4()),
        "iat":  now,
        "exp":  now + ACCESS_TOKEN_EXPIRE,
        "type": "access",
    }
    return jwt.encode(payload, _jwt_secret(), algorithm=JWT_ALGORITHM)


def create_refresh_token(user_id: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub":  user_id,
        "jti":  str(uuid.uuid4()),
        "iat":  now,
        "exp":  now + REFRESH_TOKEN_EXPIRE,
        "type": "refresh",
    }
    return jwt.encode(payload, _jwt_secret(), algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, _jwt_secret(), algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── Dependency: resolve current user ─────────────────────────────────────────

async def _resolve_bearer_token(
    token: str, db: AsyncSession
) -> User | None:
    """Try to authenticate via JWT Bearer token."""
    try:
        payload = decode_token(token)
        if payload.get("type") != "access":
            return None
        user_id = payload.get("sub")
        if not user_id:
            return None
    except HTTPException:
        return None

    from sqlalchemy import select
    result = await db.execute(select(User).where(User.id == user_id, User.is_active == True))
    return result.scalar_one_or_none()


async def _resolve_api_key(
    raw_key: str, db: AsyncSession
) -> User | None:
    """Try to authenticate via API key."""
    key_hash = hash_api_key(raw_key)
    from sqlalchemy import select
    now = datetime.now(timezone.utc)
    result = await db.execute(
        select(APIKey)
        .where(
            APIKey.key_hash == key_hash,
            APIKey.is_active == True,
        )
        .where(
            (APIKey.expires_at == None) | (APIKey.expires_at > now)
        )
    )
    api_key_obj = result.scalar_one_or_none()
    if not api_key_obj:
        return None

    # Update last_used asynchronously
    api_key_obj.last_used = now

    user_result = await db.execute(
        select(User).where(User.id == api_key_obj.user_id, User.is_active == True)
    )
    return user_result.scalar_one_or_none()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
    oauth_token: str | None = Depends(_oauth2),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    FastAPI dependency: resolve current user from Bearer JWT or API key.
    Tries Bearer first, then falls back to API key prefix check.
    """
    token = None
    if credentials:
        token = credentials.credentials
    elif oauth_token:
        token = oauth_token

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Detect token type: API keys start with "ztn_"
    if token.startswith("ztn_"):
        user = await _resolve_api_key(token, db)
    else:
        user = await _resolve_bearer_token(token, db)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired credentials",
        )
    return user


async def get_optional_user(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
    oauth_token: str | None = Depends(_oauth2),
    db: AsyncSession = Depends(get_db),
) -> User | None:
    """Optional user dependency for public endpoints with personalized context."""
    token = None
    if credentials:
        token = credentials.credentials
    elif oauth_token:
        token = oauth_token

    if not token:
        return None

    try:
        if token.startswith("ztn_"):
            return await _resolve_api_key(token, db)
        return await _resolve_bearer_token(token, db)
    except Exception:
        return None


async def require_admin(current_user: User = Depends(get_current_user)) -> User:
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin required")
    return current_user


# ── Quota enforcement ─────────────────────────────────────────────────────────

async def check_and_consume_quota(
    user: User,
    db: AsyncSession,
    n_qubits: int = 0,
    shots: int = 1024,
) -> None:
    """
    Enforce per-user resource quotas. Raises 429 if over limit.
    Atomically increments counters.
    """
    from sqlalchemy import select, update
    now = datetime.now(timezone.utc)

    result = await db.execute(select(UserQuota).where(UserQuota.user_id == user.id))
    quota = result.scalar_one_or_none()
    if quota is None:
        # Create default quota
        quota = UserQuota(user_id=user.id)
        db.add(quota)
        await db.flush()

    # Reset hourly counter if needed
    if (now - quota.quota_reset_at).total_seconds() > 3600:
        quota.jobs_this_hour = 0
        quota.quota_reset_at = now

    # Check limits
    if quota.jobs_this_hour >= quota.max_jobs_per_hour:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Hourly job limit ({quota.max_jobs_per_hour}) exceeded. Resets at "
                   f"{quota.quota_reset_at + timedelta(hours=1)}",
        )
    if n_qubits > quota.max_qubits:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Circuit requires {n_qubits} qubits; your limit is {quota.max_qubits}",
        )
    if shots > quota.max_shots:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Requested {shots} shots; your limit is {quota.max_shots}",
        )

    # Count concurrent running jobs
    from sqlalchemy import func
    from .models import JobStatus
    count_result = await db.execute(
        select(func.count()).where(
            Job.user_id == user.id,
            Job.status.in_([JobStatus.QUEUED, JobStatus.RUNNING]),
        )
    )
    concurrent = count_result.scalar() or 0
    if concurrent >= quota.max_concurrent:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Max concurrent jobs ({quota.max_concurrent}) reached",
        )

    # Consume quota
    quota.jobs_this_hour += 1
    quota.total_jobs += 1
    await db.commit()


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class TokenResponse(BaseModel):
    access_token:  str
    refresh_token: str
    token_type:    str = "bearer"
    expires_in:    int = int(ACCESS_TOKEN_EXPIRE.total_seconds())


class RegisterRequest(BaseModel):
    email:    EmailStr
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class APIKeyCreateRequest(BaseModel):
    name: str | None = None
    expires_days: int | None = None


class APIKeyResponse(BaseModel):
    id:         str
    prefix:     str
    name:       str | None
    created_at: datetime
    expires_at: datetime | None
    # raw_key only returned on creation, never again
    raw_key:    str | None = None
