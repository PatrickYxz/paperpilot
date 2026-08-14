"""Conversation route composition.

The root router owns the public `/api/conversations` prefix and composes the
CRUD, Message, and rollback child routers.  It contains no database queries or
request handling so route ownership stays visible at the module boundary.
"""
from __future__ import annotations

from fastapi import APIRouter

from paperpilot.deep_reading.runner import DeepReadingRunner
from paperpilot.web.routes.auth import RequireUser
from paperpilot.web.routes.conversations.crud import (
    build_crud_router,
    register_collection_routes,
)
from paperpilot.web.routes.conversations.messages import build_message_router
from paperpilot.web.routes.conversations.rollback import build_rollback_router
from paperpilot.web.routes.papers import PaperSearch
from paperpilot.web.task_executor import TaskExecutorLike
from paperpilot.web.task_store import TaskStore


def build_conversation_router(
    *,
    store: TaskStore,
    executor: TaskExecutorLike,
    require_user: RequireUser,
    deep_reading_runner: DeepReadingRunner,
    paper_search: PaperSearch,
    overload_retry_after_seconds: int,
) -> APIRouter:
    router = APIRouter(prefix="/api/conversations")
    register_collection_routes(
        router,
        store=store,
        require_user=require_user,
        paper_search=paper_search,
    )
    router.include_router(
        build_crud_router(
            store=store,
            require_user=require_user,
            paper_search=paper_search,
        )
    )
    router.include_router(
        build_message_router(
            store=store,
            executor=executor,
            require_user=require_user,
            overload_retry_after_seconds=overload_retry_after_seconds,
        )
    )
    router.include_router(
        build_rollback_router(
            store=store,
            require_user=require_user,
            deep_reading_runner=deep_reading_runner,
        )
    )
    return router
