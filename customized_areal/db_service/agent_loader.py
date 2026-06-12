"""
AgentLoader - Unified agent data loading service.

Provides AgentConfig/AgentData models and a loader to fetch agents with a
consistent schema and unified caching behavior.
"""

import logging
import uuid as _uuid
from dataclasses import dataclass, field
from typing import Any

from customized_areal.db_service.connection import DBConnection

logger = logging.getLogger("AgentLoader")


@dataclass
class AgentConfig:
    """Agent configuration (nested under 'config' key)."""

    system_prompt: str = ""
    model: str | None = None
    tools: dict[str, Any] = field(default_factory=lambda: {"builtin": [], "mcp": []})
    triggers: list[dict[str, Any]] = field(default_factory=list)
    context_manager_type: str = "vanilla"
    max_iterations: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_prompt": self.system_prompt,
            "model": self.model,
            "tools": self.tools,
            "triggers": self.triggers,
            "context_manager_type": self.context_manager_type,
            "max_iterations": self.max_iterations,
        }


@dataclass
class AgentData:
    """
    Complete agent data including configuration.

    This is the single source of truth for agent representation.
    """

    # Core fields from agents table
    agent_id: str
    name: str
    description: str | None
    account_id: str
    is_default: bool
    is_public: bool
    tags: list
    icon_name: str | None
    icon_color: str | None
    icon_background: str | None
    created_at: str
    updated_at: str
    current_version_id: str | None
    version_count: int
    metadata: dict[str, Any] | None

    # Nested configuration
    config: AgentConfig = field(default_factory=AgentConfig)

    # Version info
    version_name: str | None = None
    version_number: int | None = None
    version_created_at: str | None = None
    version_updated_at: str | None = None
    version_created_by: str | None = None

    # New metadata fields
    id: str | None = None  # "river", "workflow", etc.
    role: str | None = None  # "main" | "sub" | None
    config_loaded: bool = False
    restrictions: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        result = {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "account_id": self.account_id,
            "is_default": self.is_default,
            "is_public": self.is_public,
            "tags": self.tags,
            "icon_name": self.icon_name,
            "icon_color": self.icon_color,
            "icon_background": self.icon_background,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "current_version_id": self.current_version_id,
            "version_count": self.version_count,
            "metadata": self.metadata,
        }

        if self.config_loaded:
            result["config"] = self.config.to_dict()
            result["version_name"] = self.version_name
            result["restrictions"] = self.restrictions

            if self.version_number is not None:
                result["current_version"] = {
                    "version_id": self.current_version_id,
                    "version_number": self.version_number,
                    "version_name": self.version_name,
                    "created_at": self.version_created_at,
                    "updated_at": self.version_updated_at,
                    "created_by": self.version_created_by,
                }
        else:
            result["config"] = None

        return result


class AgentLoader:
    """
    Unified agent loading service.

    Handles all agent data loading with consistent behavior:
    - Single agent: loads full config
    - List operations: loads metadata only (fast)
    - Batch loading: efficient version fetching
    """

    def __init__(self, db: DBConnection | None = None):
        self.db = db or DBConnection()

    async def resolve_agent_id(self, identifier: str, account_id: str) -> str | None:
        """Resolve an agent_id (builtin ID or UUID) to a DB agent UUID.

        - If identifier is a valid UUID, return it directly.
        - Otherwise, look up via agents table metadata.
        """
        try:
            _uuid.UUID(identifier)
            return identifier
        except ValueError:
            pass

        # Try to find by metadata.id (builtin identifier)
        client = await self.db.client
        result = (
            await client.table("agents")
            .select("agent_id")
            .eq("account_id", account_id)
            .eq("metadata->>id", identifier)
            .execute()
        )
        if result.data:
            return result.data[0]["agent_id"]
        return None

    async def load_agent(
        self,
        agent_id: str,
        user_id: str,
        load_config: bool = True,
        skip_cache: bool = False,
    ) -> AgentData:
        """
        Load a single agent with full configuration.

        Args:
            agent_id: Agent ID to load
            user_id: User ID for authorization
            load_config: Whether to load full version configuration
            skip_cache: If True, bypass cache (for cache warm-up)

        Returns:
            AgentData with complete information

        Raises:
            ValueError: If agent not found or access denied
        """
        client = await self.db.client

        # Fetch agent metadata
        result = (
            await client.table("agents").select("*").eq("agent_id", agent_id).execute()
        )

        if not result.data:
            raise ValueError(f"Agent {agent_id} not found")

        agent_row = result.data[0]

        # Check access
        if agent_row["account_id"] != user_id and not agent_row.get("is_public", False):
            raise ValueError(f"Access denied to agent {agent_id}")

        # Create base AgentData
        agent_data = self._row_to_agent_data(agent_row)

        # Load configuration if requested
        if load_config and agent_data.current_version_id:
            await self._load_agent_config(agent_data, user_id)

        return agent_data

    async def load_agents_list(
        self, agent_rows: list, load_config: bool = False
    ) -> list[AgentData]:
        """
        Load multiple agents efficiently.

        Args:
            agent_rows: List of agent database rows
            load_config: Whether to batch-load configurations

        Returns:
            List of AgentData objects
        """
        agents = [self._row_to_agent_data(row) for row in agent_rows]

        if load_config:
            await self._batch_load_configs(agents)

        return agents

    async def load_template(
        self, template_row: dict[str, Any], fetch_creator_name: bool = False
    ) -> AgentData:
        """
        Load a template as AgentData.

        Templates are basically agents with pre-configured settings.

        Args:
            template_row: Template database row
            fetch_creator_name: Whether to fetch creator name

        Returns:
            AgentData representing the template
        """
        metadata = template_row.get("metadata", {}) or {}

        # Fetch creator name if requested
        creator_name = None
        if fetch_creator_name and template_row.get("creator_id"):
            try:
                client = await self.db.client
                creator_result = (
                    await client.schema("basejump")
                    .from_("accounts")
                    .select("name, slug")
                    .eq("id", template_row["creator_id"])
                    .single()
                    .execute()
                )
                if creator_result.data:
                    creator_name = creator_result.data.get(
                        "name"
                    ) or creator_result.data.get("slug")
            except Exception as e:
                logger.warning(f"Failed to fetch creator name: {e}")

        # Update metadata
        metadata["is_template"] = True
        if creator_name:
            metadata["creator_name"] = creator_name

        # Create AgentData from template
        agent_data = AgentData(
            agent_id=template_row.get("template_id", ""),
            name=template_row.get("name", ""),
            description=template_row.get("description"),
            account_id=template_row.get("creator_id", ""),
            is_default=False,
            is_public=template_row.get("is_public", False),
            tags=template_row.get("tags", []),
            icon_name=template_row.get("icon_name"),
            icon_color=template_row.get("icon_color"),
            icon_background=template_row.get("icon_background"),
            created_at=template_row.get("created_at", ""),
            updated_at=template_row.get("updated_at"),
            current_version_id=None,
            version_count=0,
            metadata=metadata,
            config=AgentConfig(
                system_prompt=template_row.get("system_prompt", ""),
                model=metadata.get("model"),
                tools={
                    "builtin": template_row.get("builtin_tools", []),
                    "mcp": template_row.get("mcp_requirements", []),
                },
                triggers=[],
            ),
            version_name="template",
            id=None,
            role=None,
            config_loaded=True,
            restrictions={},
        )

        return agent_data

    def _dict_to_agent_data(self, data: dict[str, Any]) -> AgentData:
        """Convert cached dict back to AgentData."""
        current_version = data.get("current_version", {}) or {}
        config_dict = data.get("config") or {}
        metadata = data.get("metadata", {})

        return AgentData(
            agent_id=data["agent_id"],
            name=data["name"],
            description=data.get("description"),
            account_id=data["account_id"],
            is_default=data.get("is_default", False),
            is_public=data.get("is_public", False),
            tags=data.get("tags", []),
            icon_name=data.get("icon_name"),
            icon_color=data.get("icon_color"),
            icon_background=data.get("icon_background"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            current_version_id=data.get("current_version_id"),
            version_count=data.get("version_count", 1),
            metadata=metadata,
            config=AgentConfig(
                system_prompt=config_dict.get("system_prompt", ""),
                model=config_dict.get("model"),
                tools=config_dict.get("tools", {"builtin": [], "mcp": []}),
                triggers=config_dict.get("triggers", []),
                context_manager_type=config_dict.get("context_manager_type", "vanilla"),
                max_iterations=config_dict.get("max_iterations"),
            ),
            version_name=data.get("version_name")
            or current_version.get("version_name"),
            version_number=current_version.get("version_number"),
            version_created_at=current_version.get("created_at"),
            version_updated_at=current_version.get("updated_at"),
            version_created_by=current_version.get("created_by"),
            id=metadata.get("id"),
            role=metadata.get("role"),
            config_loaded=True,
            restrictions=metadata.get("restrictions", {}),
        )

    def _row_to_agent_data(self, row: dict[str, Any]) -> AgentData:
        """Convert database row to AgentData."""
        metadata = row.get("metadata", {}) or {}

        return AgentData(
            agent_id=row["agent_id"],
            name=row["name"],
            description=row.get("description"),
            account_id=row["account_id"],
            is_default=row.get("is_default", False),
            is_public=row.get("is_public", False),
            tags=row.get("tags", []),
            icon_name=row.get("icon_name"),
            icon_color=row.get("icon_color"),
            icon_background=row.get("icon_background"),
            created_at=row["created_at"],
            updated_at=row.get("updated_at", row["created_at"]),
            current_version_id=row.get("current_version_id"),
            version_count=row.get("version_count", 1),
            metadata=metadata,
            id=metadata.get("id"),
            role=metadata.get("role"),
            config_loaded=False,
        )

    async def _load_agent_config(self, agent: AgentData, user_id: str):
        """Load full configuration for a single agent."""
        if not agent.current_version_id:
            raise ValueError(
                f"Agent {agent.agent_id} has no current_version_id, cannot load config"
            )

        client = await self.db.client

        # Fetch version configuration directly from agent_versions table
        result = (
            await client.table("agent_versions")
            .select("*")
            .eq("version_id", agent.current_version_id)
            .eq("agent_id", agent.agent_id)
            .execute()
        )

        if not result.data:
            raise ValueError(
                f"Version {agent.current_version_id} not found for agent {agent.agent_id}"
            )

        version_row = result.data[0]
        self._apply_version_config(agent, version_row)
        agent.config_loaded = True

    def _apply_version_config(self, agent: AgentData, version_row: dict[str, Any]):
        """Apply version configuration to agent."""
        config = version_row.get("config") or {}
        tools = config.get("tools", {})

        agent.config = AgentConfig(
            system_prompt=config.get("system_prompt", ""),
            model=config.get("model"),
            tools={
                "builtin": tools.get("builtin", []),
                "mcp": tools.get("mcp", []),
            },
            triggers=config.get("triggers", []),
            context_manager_type=config.get("context_manager_type", "vanilla"),
            max_iterations=config.get("max_iterations"),
        )
        agent.version_name = version_row.get("version_name", "v1")
        agent.version_number = version_row.get("version_number")
        agent.version_created_at = version_row.get("created_at")
        agent.version_updated_at = version_row.get("updated_at")
        agent.version_created_by = version_row.get("created_by")
        agent.restrictions = {}

    async def _batch_load_configs(self, agents: list[AgentData]):
        """Batch load configurations for multiple agents."""
        # Get all version IDs for agents
        version_ids = [a.current_version_id for a in agents if a.current_version_id]

        if not version_ids:
            return

        try:
            client = await self.db.client

            # Fetch all versions in one query
            result = (
                await client.table("agent_versions")
                .select("*")
                .in_("version_id", version_ids)
                .execute()
            )

            # Create version map
            version_map = {}
            for version_row in result.data or []:
                version_map[version_row["version_id"]] = version_row

            # Apply configs
            for agent in agents:
                if agent.current_version_id and agent.current_version_id in version_map:
                    self._apply_version_config(
                        agent, version_map[agent.current_version_id]
                    )
                    agent.config_loaded = True

        except Exception as e:
            logger.warning(f"Failed to batch load agent configs: {e}")


_loader: AgentLoader | None = None


async def get_agent_loader() -> AgentLoader:
    """Get the global AgentLoader instance."""
    global _loader
    if _loader is None:
        _loader = AgentLoader()
    return _loader


__all__ = [
    "AgentConfig",
    "AgentData",
    "AgentLoader",
    "get_agent_loader",
]
