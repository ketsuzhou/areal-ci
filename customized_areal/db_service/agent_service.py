"""Agent service with CRUD operations for TPFC backend.

This is a simplified version of the agent service for use in the AReaL training workflow.
It provides the minimal functionality needed to create and manage agents.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .pagination import PaginatedResponse, PaginationParams, PaginationService
from .schemas import AgentConfig, AgentCreateRequest, AgentResponse, AgentUpdateRequest


@dataclass
class AgentFilters:
    """Filters for agent list queries."""

    search: str | None = None
    has_default: bool | None = None
    has_mcp_tools: bool | None = None
    has_builtin_tools: bool | None = None
    tools: list[str] = field(default_factory=list)
    content_type: str | None = None
    sort_by: str = "created_at"
    sort_order: str = "desc"


class AgentService:
    """Service for agent CRUD operations."""

    def __init__(self, db_client):
        self.db = db_client

    async def create_agent(
        self, user_id: str, data: AgentCreateRequest
    ) -> AgentResponse:
        """Create a new agent with initial version."""
        # If setting as default, clear other defaults
        if data.is_default:
            await self._clear_default_agents(user_id)

        agent_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()

        # Build agent record
        agent_record = {
            "agent_id": agent_id,
            "account_id": user_id,
            "name": data.name,
            "icon_name": data.icon_name or "bot",
            "icon_color": data.icon_color or "#000000",
            "icon_background": data.icon_background or "#F3F4F6",
            "is_default": data.is_default or False,
            "is_public": False,
            "version_count": 1,
            "created_at": now,
            "updated_at": now,
            "current_version_id": None,
            "metadata": {},
        }

        # Insert agent
        result = await self.db.table("agents").insert(agent_record).execute()
        if not result.data:
            raise RuntimeError("Failed to create agent")

        # Create initial version if config provided
        if data.config:
            version_id = await self._create_initial_version(
                agent_id=agent_id,
                user_id=user_id,
                config=data.config,
            )
            agent_record["current_version_id"] = version_id

            # Update agent with version
            await (
                self.db.table("agents")
                .update({"current_version_id": version_id})
                .eq("agent_id", agent_id)
                .execute()
            )

        return self._to_agent_response(agent_record)

    async def update_agent(
        self, agent_id: str, user_id: str, data: AgentUpdateRequest
    ) -> AgentResponse:
        """Update an agent with optional version creation."""
        # Get existing agent
        existing = await self._get_agent_record(agent_id, user_id)
        if not existing:
            raise ValueError(f"Agent not found: {agent_id}")

        # Handle is_default change
        if data.is_default is not None and data.is_default:
            await self._clear_default_agents(user_id, exclude_agent_id=agent_id)

        # Build update data
        update_data: dict[str, Any] = {"updated_at": datetime.now(UTC).isoformat()}
        if data.name is not None:
            update_data["name"] = data.name
        if data.description is not None:
            update_data["description"] = data.description
        if data.is_default is not None:
            update_data["is_default"] = data.is_default
        if data.icon_name is not None:
            update_data["icon_name"] = data.icon_name
        if data.icon_color is not None:
            update_data["icon_color"] = data.icon_color
        if data.icon_background is not None:
            update_data["icon_background"] = data.icon_background

        # Create new version if config changed
        if data.config:
            version_id = await self._create_new_version(
                agent_id=agent_id,
                user_id=user_id,
                config=data.config,
                current_version_id=existing.get("current_version_id"),
            )
            update_data["current_version_id"] = version_id
            update_data["version_count"] = existing.get("version_count", 1) + 1

        # Update agent
        if update_data:
            result = (
                await self.db.table("agents")
                .update(update_data)
                .eq("agent_id", agent_id)
                .eq("account_id", user_id)
                .execute()
            )
            if not result.data:
                raise RuntimeError("Failed to update agent")

            # Merge updates into existing record for response
            existing.update(update_data)

        return self._to_agent_response(existing)

    async def delete_agent(self, agent_id: str, user_id: str) -> dict[str, str]:
        """Delete an agent."""
        # Verify agent exists and belongs to user
        agent = await self._get_agent_record(agent_id, user_id)
        if not agent:
            raise ValueError(f"Agent not found: {agent_id}")

        if agent.get("is_default"):
            raise ValueError("Cannot delete default agent")

        # Delete agent (cascade will handle versions)
        result = (
            await self.db.table("agents")
            .delete()
            .eq("agent_id", agent_id)
            .eq("account_id", user_id)
            .execute()
        )

        if not result.data:
            raise RuntimeError("Failed to delete agent")

        return {"message": "Agent deleted successfully"}

    async def get_agent(self, agent_id: str, user_id: str) -> AgentResponse:
        """Get a single agent with full configuration."""
        agent = await self._get_agent_record(agent_id, user_id)
        if not agent:
            raise ValueError(f"Agent not found: {agent_id}")
        return self._to_agent_response(agent)

    async def get_agents_paginated(
        self, user_id: str, pagination_params: PaginationParams, filters: AgentFilters
    ) -> PaginatedResponse[AgentResponse]:
        """Get paginated list of agents with filters."""
        # Build base query
        query = self.db.table("agents").select("*").eq("account_id", user_id)

        # Apply search filter
        if filters.search:
            search_term = f"%{filters.search}%"
            query = query.or_(
                f"name.ilike.{search_term},description.ilike.{search_term}"
            )

        # Apply has_default filter
        if filters.has_default is not None:
            query = query.eq("is_default", filters.has_default)

        # Apply sorting
        sort_column = (
            filters.sort_by
            if filters.sort_by in ["name", "created_at", "updated_at"]
            else "created_at"
        )
        query = query.order(sort_column, desc=(filters.sort_order == "desc"))

        # Build count query
        count_query = (
            self.db.table("agents").select("*", count="exact").eq("account_id", user_id)
        )
        if filters.search:
            search_term = f"%{filters.search}%"
            count_query = count_query.or_(
                f"name.ilike.{search_term},description.ilike.{search_term}"
            )
        if filters.has_default is not None:
            count_query = count_query.eq("is_default", filters.has_default)

        # Execute pagination
        paginated_result = await PaginationService.paginate_database_query(
            base_query=query, params=pagination_params, count_query=count_query
        )

        # Transform to AgentResponse
        agents = [self._to_agent_response(row) for row in paginated_result.data]

        return PaginatedResponse(
            data=agents,
            pagination=paginated_result.pagination,
        )

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    async def _get_agent_record(
        self, agent_id: str, user_id: str
    ) -> dict[str, Any] | None:
        """Get raw agent record from database."""
        result = (
            await self.db.table("agents")
            .select("*")
            .eq("agent_id", agent_id)
            .eq("account_id", user_id)
            .maybe_single()
            .execute()
        )
        return result.data if result else None

    async def _clear_default_agents(
        self, user_id: str, exclude_agent_id: str | None = None
    ) -> None:
        """Clear is_default flag on all user agents except excluded one."""
        query = (
            self.db.table("agents")
            .update({"is_default": False})
            .eq("account_id", user_id)
            .eq("is_default", True)
        )
        if exclude_agent_id:
            query = query.neq("agent_id", exclude_agent_id)
        await query.execute()

    async def _create_initial_version(
        self,
        agent_id: str,
        user_id: str,
        config: AgentConfig,
    ) -> str:
        """Create initial version for a new agent."""
        version_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()

        config_dict = self._config_to_dict(config)

        version_record = {
            "version_id": version_id,
            "agent_id": agent_id,
            "version_number": 1,
            "version_name": "v1",
            "config": config_dict,
            "is_active": True,
            "created_at": now,
            "updated_at": now,
            "created_by": user_id,
        }

        result = await self.db.table("agent_versions").insert(version_record).execute()
        if not result.data:
            raise RuntimeError("Failed to create initial version")

        return version_id

    async def _create_new_version(
        self,
        agent_id: str,
        user_id: str,
        config: AgentConfig,
        current_version_id: str | None,
    ) -> str:
        """Create a new agent version."""
        # Get current version number
        version_number = 1
        if current_version_id:
            result = (
                await self.db.table("agent_versions")
                .select("version_number")
                .eq("version_id", current_version_id)
                .maybe_single()
                .execute()
            )
            if result.data:
                version_number = result.data.get("version_number", 0) + 1

        version_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()

        config_dict = self._config_to_dict(config)

        version_record = {
            "version_id": version_id,
            "agent_id": agent_id,
            "version_number": version_number,
            "version_name": f"v{version_number}",
            "config": config_dict,
            "is_active": True,
            "created_at": now,
            "updated_at": now,
            "created_by": user_id,
        }

        result = await self.db.table("agent_versions").insert(version_record).execute()
        if not result.data:
            raise RuntimeError("Failed to create new version")

        return version_id

    def _config_to_dict(self, config: AgentConfig) -> dict[str, Any]:
        """Convert AgentConfig to dictionary."""
        return {
            "system_prompt": config.system_prompt,
            "model": config.model,
            "tools": {
                "builtin": config.tools.builtin if config.tools else [],
                "mcp": config.tools.mcp if config.tools else [],
            },
            "triggers": config.triggers,
            "context_manager_type": config.context_manager_type,
            "max_iterations": config.max_iterations,
        }

    def _to_agent_response(self, record: dict[str, Any]) -> AgentResponse:
        """Convert database record to AgentResponse."""
        return AgentResponse(
            agent_id=record["agent_id"],
            name=record["name"],
            description=record.get("description"),
            config=None,  # Loaded separately if needed
            is_default=record.get("is_default", False),
            is_public=record.get("is_public", False),
            tags=record.get("tags", []),
            icon_name=record.get("icon_name"),
            icon_color=record.get("icon_color"),
            icon_background=record.get("icon_background"),
            created_at=record["created_at"],
            updated_at=record.get("updated_at"),
            current_version_id=record.get("current_version_id"),
            version_count=record.get("version_count", 1),
            current_version=None,
            metadata=record.get("metadata"),
            account_id=record.get("account_id"),
        )


def get_agent_service(db_client) -> AgentService:
    """Factory function to create AgentService instance."""
    return AgentService(db_client)
