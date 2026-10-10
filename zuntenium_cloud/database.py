# zuntenium_cloud/database.py
"""
Async database session management using SQLAlchemy + asyncpg.
Connection pool configured for production (25 + 10 overflow).
"""

from __future__ import annotations

import os
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

_DATABASE_URL = os.environ.get(
    "ZUNTENIUM_DATABASE_URL",
    "postgresql+asyncpg://zuntenium:zuntenium@localhost:5432/zuntenium",
)

# Use NullPool in test environments to avoid connection leaks
_pool_cls = NullPool if os.environ.get("TESTING") else None

_engine_kwargs = {
    "echo": bool(os.environ.get("DB_ECHO")),
    "pool_pre_ping": True,
}

if _pool_cls:
    _engine_kwargs["poolclass"] = _pool_cls
else:
    _engine_kwargs.update({
        "pool_size": 25,
        "max_overflow": 10,
        "pool_timeout": 30,
        "pool_recycle": 1800,
    })

engine = create_async_engine(
    _DATABASE_URL,
    **_engine_kwargs,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: yield an async DB session, commit on success."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Create all tables (run once at startup or via Alembic migrations)."""
    from .models import Base
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
