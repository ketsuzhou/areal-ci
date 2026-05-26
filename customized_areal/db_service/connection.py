"""Centralized database connection management using Supabase."""

from __future__ import annotations

import os
import threading

from httpx import AsyncClient as AsyncHttpxClient
from httpx import Client as SyncHttpxClient
from httpx import Timeout
from supabase import AsyncClient, Client, create_async_client, create_client
from supabase.lib.client_options import AsyncClientOptions, SyncClientOptions


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

        # Increased timeouts for unstable VPN connections and large queries
        httpx_client = SyncHttpxClient(
            timeout=Timeout(connect=30.0, read=120.0, write=30.0, pool=30.0)
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

        # Increased timeouts for unstable VPN connections and large queries
        httpx_client = AsyncHttpxClient(
            timeout=Timeout(connect=30.0, read=120.0, write=30.0, pool=30.0)
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
