"""Centralized database connection management using Supabase."""

from __future__ import annotations

import logging
import os
import threading
from urllib.parse import urlparse

from httpx import AsyncClient as AsyncHttpxClient
from httpx import Client as SyncHttpxClient
from httpx import Limits, Timeout
from supabase import AsyncClient, Client, create_async_client, create_client
from supabase.lib.client_options import AsyncClientOptions, SyncClientOptions

logger = logging.getLogger(__name__)

_SUPABASE_CONNECT_TIMEOUT = 30.0
_SUPABASE_READ_TIMEOUT = 120.0
_SUPABASE_WRITE_TIMEOUT = 30.0
_SUPABASE_POOL_TIMEOUT = 60.0
_SUPABASE_MAX_CONNECTIONS = 100
_SUPABASE_MAX_KEEPALIVE_CONNECTIONS = 50
_SUPABASE_KEEPALIVE_EXPIRY = 30


def _describe_supabase_url(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return "<invalid>"
    return f"{parsed.scheme}://{parsed.netloc}"


def _build_sync_supabase_httpx_client() -> SyncHttpxClient:
    """Build a Supabase HTTP client that ignores ambient proxy env vars."""
    return SyncHttpxClient(
        timeout=Timeout(
            connect=_SUPABASE_CONNECT_TIMEOUT,
            read=_SUPABASE_READ_TIMEOUT,
            write=_SUPABASE_WRITE_TIMEOUT,
            pool=_SUPABASE_POOL_TIMEOUT,
        ),
        limits=Limits(
            max_connections=_SUPABASE_MAX_CONNECTIONS,
            max_keepalive_connections=_SUPABASE_MAX_KEEPALIVE_CONNECTIONS,
            keepalive_expiry=_SUPABASE_KEEPALIVE_EXPIRY,
        ),
        trust_env=False,
    )


def _build_async_supabase_httpx_client(
    *,
    max_connections: int = 100,
    max_keepalive_connections: int = 50,
) -> AsyncHttpxClient:
    """Build a Supabase HTTP client that ignores ambient proxy env vars."""
    return AsyncHttpxClient(
        timeout=Timeout(
            connect=_SUPABASE_CONNECT_TIMEOUT,
            read=_SUPABASE_READ_TIMEOUT,
            write=_SUPABASE_WRITE_TIMEOUT,
            pool=_SUPABASE_POOL_TIMEOUT,
        ),
        limits=Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
            keepalive_expiry=_SUPABASE_KEEPALIVE_EXPIRY,
        ),
        trust_env=False,
    )


class SyncDBConnection:
    """Thread-safe singleton for synchronous Supabase client.

    Used for synchronous database operations in contexts where async is not
    available (e.g., synchronous file operations or background tasks).
    """

    _instance: SyncDBConnection | None = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
                    cls._instance._client: Client | None = None
        return cls._instance

    def __init__(self):
        pass

    def initialize(self):
        """Initialize the sync database connection."""
        if self._initialized:
            return

        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get(
            "SUPABASE_ANON_KEY"
        )

        if not supabase_url or not supabase_key:
            raise RuntimeError(
                "SUPABASE_URL and a key (SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY) "
                "environment variables must be set."
            )

        # Pool sized for general-purpose (non-rollout) usage
        httpx_client = _build_sync_supabase_httpx_client()
        logger.info(
            "Initializing sync Supabase client: url=%s trust_env=%s max_connections=%s "
            "max_keepalive=%s keepalive_expiry=%s timeout=(connect=%s read=%s write=%s pool=%s)",
            _describe_supabase_url(supabase_url),
            False,
            _SUPABASE_MAX_CONNECTIONS,
            _SUPABASE_MAX_KEEPALIVE_CONNECTIONS,
            _SUPABASE_KEEPALIVE_EXPIRY,
            _SUPABASE_CONNECT_TIMEOUT,
            _SUPABASE_READ_TIMEOUT,
            _SUPABASE_WRITE_TIMEOUT,
            _SUPABASE_POOL_TIMEOUT,
        )
        options = SyncClientOptions(httpx_client=httpx_client)
        self._client = create_client(supabase_url, supabase_key, options)
        self._initialized = True

    @classmethod
    def disconnect(cls):
        """Disconnect from the database and close httpx client."""
        with cls._lock:
            if cls._instance and cls._instance._client:
                try:
                    if hasattr(cls._instance._client, "options") and hasattr(
                        cls._instance._client.options, "httpx_client"
                    ):
                        cls._instance._client.options.httpx_client.close()
                except Exception:
                    pass
                finally:
                    cls._instance._initialized = False
                    cls._instance._client = None

    @property
    def client(self) -> Client:
        """Get the sync Supabase client instance."""
        if not self._initialized:
            self.initialize()
        if not self._client:
            raise RuntimeError("Sync database not initialized")
        return self._client


class DBConnection:
    """Thread-safe singleton database connection manager using Supabase.

    This class provides async database connection management with automatic
    initialization and proper connection pooling.

    Usage:
        db = DBConnection()
        client = await db.get_client()
        result = await client.table("tasks").select("*").execute()
    """

    _instance: DBConnection | None = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                # Double-check locking pattern
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
                    cls._instance._client: AsyncClient | None = None
        return cls._instance

    def __init__(self):
        """No initialization needed in __init__ as it's handled in __new__"""
        pass

    async def initialize(self):
        """Initialize the database connection."""
        if self._initialized:
            return

        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get(
            "SUPABASE_ANON_KEY"
        )

        if not supabase_url or not supabase_key:
            raise RuntimeError(
                "SUPABASE_URL and a key (SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY) "
                "environment variables must be set."
            )

        # Pool sized for general-purpose (non-rollout) usage
        httpx_client = _build_async_supabase_httpx_client()
        logger.info(
            "Initializing async Supabase client: url=%s trust_env=%s max_connections=%s "
            "max_keepalive=%s keepalive_expiry=%s timeout=(connect=%s read=%s write=%s pool=%s)",
            _describe_supabase_url(supabase_url),
            False,
            _SUPABASE_MAX_CONNECTIONS,
            _SUPABASE_MAX_KEEPALIVE_CONNECTIONS,
            _SUPABASE_KEEPALIVE_EXPIRY,
            _SUPABASE_CONNECT_TIMEOUT,
            _SUPABASE_READ_TIMEOUT,
            _SUPABASE_WRITE_TIMEOUT,
            _SUPABASE_POOL_TIMEOUT,
        )
        options = AsyncClientOptions(httpx_client=httpx_client)
        self._client = await create_async_client(
            supabase_url,
            supabase_key,
            options,
        )
        self._initialized = True

    @classmethod
    async def disconnect(cls):
        """Disconnect from the database and close httpx client."""
        if cls._instance and cls._instance._client:
            try:
                if hasattr(cls._instance._client, "options") and hasattr(
                    cls._instance._client.options, "httpx_client"
                ):
                    await cls._instance._client.options.httpx_client.aclose()
            except Exception:
                pass
            finally:
                cls._instance._initialized = False
                cls._instance._client = None

    async def get_client(self) -> AsyncClient:
        """Get the Supabase client instance, initializing if needed."""
        if not self._initialized:
            await self.initialize()
        if not self._client:
            raise RuntimeError("Database not initialized")
        return self._client

    @property
    async def client(self) -> AsyncClient:
        """Get the Supabase client instance.

        Deprecated: use ``await db.get_client()`` instead.
        This async property works at runtime but is hard to mock in tests.
        """
        return await self.get_client()
