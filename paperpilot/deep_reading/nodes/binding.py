"""Business-object binding checks shared by deep-reading nodes."""
from __future__ import annotations

from langchain.messages import AnyMessage

from paperpilot.web.task_store import (
    ConversationDetail,
    MessageRecord,
    ResearchTask,
)

from ..research_agent import TaskBindingError
from ..state import DeepReadingState
from .context import DeepReadingContext


def _validate_runtime_binding(
    state: DeepReadingState,
    context: DeepReadingContext,
) -> tuple[ResearchTask, MessageRecord, ConversationDetail]:
    task_id = _required_binding_text(context.task_id, "runtime task_id")
    task = context.task_store.get_task(task_id, user_id=context.user_id)
    if task is None:
        raise TaskBindingError("runtime task is unavailable for the owner")
    if task.id != task_id or task.user_id != context.user_id:
        raise TaskBindingError("runtime task does not match the requested owner")

    detail = context.task_store.get_conversation_detail(
        context.conversation_id,
        user_id=context.user_id,
    )
    if detail is None:
        raise TaskBindingError("conversation is unavailable for the owner")
    conversation = detail.conversation
    if conversation.id != context.conversation_id:
        raise TaskBindingError(
            "conversation detail does not match runtime context"
        )
    if conversation.user_id != context.user_id:
        raise TaskBindingError("conversation detail does not match runtime owner")
    if task.conversation_id != conversation.id:
        raise TaskBindingError("runtime task belongs to another conversation")
    if task.base_checkpoint_id != context.base_checkpoint_id:
        raise TaskBindingError(
            "runtime task base checkpoint does not match context"
        )
    state_task_id = _required_binding_text(
        state.get("current_task_id"),
        "current_task_id",
    )
    if state_task_id != task.id:
        raise TaskBindingError("state current_task_id does not match runtime task")

    message = context.task_store.get_task_message(task.id, "user")
    if message is None:
        raise TaskBindingError("runtime task user message is unavailable")
    if message.id != context.current_user_message_id:
        raise TaskBindingError("runtime task is bound to another user message")
    if message.role != "user" or message.status != "complete":
        raise TaskBindingError("runtime task user message is not complete")
    if message.conversation_id != context.conversation_id:
        raise TaskBindingError(
            "runtime user message belongs to another conversation"
        )
    if message.task_id != task.id:
        raise TaskBindingError("runtime user message belongs to another task")
    if task.question != message.content:
        raise TaskBindingError("runtime task question does not match user message")
    state_message_id = _required_binding_text(
        state.get("current_user_message_id"),
        "current_user_message_id",
    )
    if state_message_id != message.id:
        raise TaskBindingError(
            "state current_user_message_id does not match runtime user message"
        )

    state_messages = state.get("messages")
    if not isinstance(state_messages, list):
        raise TaskBindingError("state messages must be a list")
    matching_messages = [
        item for item in state_messages if getattr(item, "id", None) == message.id
    ]
    if len(matching_messages) != 1:
        raise TaskBindingError(
            "state must contain exactly one current user message"
        )
    current_human = matching_messages[0]
    if getattr(current_human, "type", None) != "human":
        raise TaskBindingError("state current user message must be human")
    if current_human.content != message.content:
        raise TaskBindingError(
            "state current user message content does not match business message"
        )
    human_messages = [
        item for item in state_messages if getattr(item, "type", None) == "human"
    ]
    if not human_messages or human_messages[-1] is not current_human:
        raise TaskBindingError("state current user message must be the last human")
    return task, message, detail


def _required_binding_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TaskBindingError(f"{field_name} must be a non-blank string")
    return value.strip()
