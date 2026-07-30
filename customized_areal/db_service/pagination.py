"""Pagination utilities for database queries."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class PaginationMeta(BaseModel):
    """Pagination metadata."""

    current_page: int
    page_size: int
    total_items: int
    total_pages: int
    has_next: bool
    has_previous: bool
    next_cursor: str | None = None
    previous_cursor: str | None = None


class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated response wrapper."""

    data: list[T]
    pagination: PaginationMeta


@dataclass
class PaginationParams:
    """Parameters for pagination."""

    page: int = 1
    page_size: int = 20
    cursor: str | None = None

    def __post_init__(self):
        self.page = max(1, self.page)
        self.page_size = min(max(1, self.page_size), 100)


class PaginationService:
    """Service for handling pagination logic."""

    @staticmethod
    async def paginate_with_total_count(
        items: list[T], total_count: int, params: PaginationParams
    ) -> PaginatedResponse[T]:
        """Create paginated response when you already have items and total count."""
        total_pages = max(1, math.ceil(total_count / params.page_size))

        pagination_meta = PaginationMeta(
            current_page=params.page,
            page_size=params.page_size,
            total_items=total_count,
            total_pages=total_pages,
            has_next=params.page < total_pages,
            has_previous=params.page > 1,
        )

        return PaginatedResponse(data=items, pagination=pagination_meta)

    @staticmethod
    async def paginate_database_query(
        base_query: Any,
        params: PaginationParams,
        count_query: Any | None = None,
    ) -> PaginatedResponse[dict[str, Any]]:
        """Paginate a database query."""
        if count_query:
            count_result = await count_query.execute()
            total_count = count_result.count if count_result.count else 0
        else:
            count_result = await base_query.select("*", count="exact").execute()
            total_count = count_result.count if count_result.count else 0

        if total_count == 0:
            return PaginatedResponse(
                data=[],
                pagination=PaginationMeta(
                    current_page=params.page,
                    page_size=params.page_size,
                    total_items=0,
                    total_pages=0,
                    has_next=False,
                    has_previous=False,
                ),
            )

        offset = (params.page - 1) * params.page_size
        data_query = base_query.range(offset, offset + params.page_size - 1)
        data_result = await data_query.execute()
        items = data_result.data or []

        total_pages = max(1, math.ceil(total_count / params.page_size))

        pagination_meta = PaginationMeta(
            current_page=params.page,
            page_size=params.page_size,
            total_items=total_count,
            total_pages=total_pages,
            has_next=params.page < total_pages,
            has_previous=params.page > 1,
        )

        return PaginatedResponse(data=items, pagination=pagination_meta)

    @staticmethod
    async def paginate_filtered_dataset(
        all_items: list[T], params: PaginationParams
    ) -> PaginatedResponse[T]:
        """Paginate an already filtered list of items."""
        total_count = len(all_items)

        if total_count == 0:
            return PaginatedResponse(
                data=[],
                pagination=PaginationMeta(
                    current_page=params.page,
                    page_size=params.page_size,
                    total_items=0,
                    total_pages=0,
                    has_next=False,
                    has_previous=False,
                ),
            )

        start_index = (params.page - 1) * params.page_size
        end_index = start_index + params.page_size
        page_items = all_items[start_index:end_index]

        total_pages = max(1, math.ceil(total_count / params.page_size))

        pagination_meta = PaginationMeta(
            current_page=params.page,
            page_size=params.page_size,
            total_items=total_count,
            total_pages=total_pages,
            has_next=params.page < total_pages,
            has_previous=params.page > 1,
        )

        return PaginatedResponse(data=page_items, pagination=pagination_meta)
