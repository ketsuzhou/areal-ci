"""Agent-related API models and schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AgentToolsConfig(BaseModel):
    """Tools configuration."""

    builtin: list[dict[str, Any]] = Field(default_factory=list)
    mcp: list[dict[str, Any]] = Field(default_factory=list)


class AgentConfig(BaseModel):
    """Agent configuration."""

    system_prompt: str = ""
    model: str | None = None
    tools: AgentToolsConfig = Field(default_factory=AgentToolsConfig)
    triggers: list[dict[str, Any]] = Field(default_factory=list)
    context_manager_type: str | None = "vanilla"
    max_iterations: int | None = None


class AgentCreateRequest(BaseModel):
    """Request model for creating a new agent."""

    name: str = Field(description="Display name for the agent")
    config: AgentConfig | None = Field(
        default=None,
        description="Agent configuration including system prompt, model, and tools",
    )
    is_default: bool | None = Field(
        default=False, description="Whether this is a default agent"
    )
    icon_name: str | None = Field(default=None, description="Icon name for the agent")
    icon_color: str | None = Field(default=None, description="Icon foreground color")
    icon_background: str | None = Field(
        default=None, description="Icon background color"
    )


class AgentUpdateRequest(BaseModel):
    """Request model for updating an existing agent."""

    name: str | None = Field(
        default=None, description="Updated display name for the agent"
    )
    description: str | None = Field(
        default=None, description="Updated short agent description"
    )
    config: AgentConfig | None = Field(
        default=None,
        description="Updated agent configuration including system prompt, model, and tools",
    )
    is_default: bool | None = Field(
        default=None, description="Whether this is a default agent"
    )
    icon_name: str | None = Field(
        default=None, description="Updated icon name for the agent"
    )
    icon_color: str | None = Field(
        default=None, description="Updated icon foreground color"
    )
    icon_background: str | None = Field(
        default=None, description="Updated icon background color"
    )


class AgentVersionResponse(BaseModel):
    """Response model for agent version information."""

    version_id: str
    agent_id: str
    version_number: int
    version_name: str
    config: AgentConfig
    is_active: bool
    created_at: str
    updated_at: str
    created_by: str | None = None


class AgentResponse(BaseModel):
    """Response model for agent information."""

    agent_id: str
    name: str
    description: str | None = None
    config: AgentConfig | None = None
    is_default: bool
    is_public: bool | None = False
    tags: list[str] | None = Field(default_factory=list)
    icon_name: str | None = None
    icon_color: str | None = None
    icon_background: str | None = None
    created_at: str
    updated_at: str | None = None
    current_version_id: str | None = None
    version_count: int | None = 1
    current_version: AgentVersionResponse | None = None
    metadata: dict[str, Any] | None = None
    account_id: str | None = None


class AgentsResponse(BaseModel):
    """Response model for list of agents with pagination."""

    agents: list[AgentResponse]
    pagination: dict[str, Any]


class AgentBlueprint(BaseModel):
    """Portable agent definition for import/export."""

    name: str
    description: str | None = None
    config: AgentConfig
    icon_name: str | None = None
    icon_color: str | None = None
    icon_background: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
