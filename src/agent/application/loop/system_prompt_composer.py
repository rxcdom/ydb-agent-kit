"""Compose the system message of one agent turn.

The system message is the only history item that is never trimmed, so everything
in it is paid for on every model call of the turn. It therefore carries the
prompt plus three small, size-bounded blocks, and never the content of the
memory vault: the pointer block tells the model that ``recall`` exists, and the
model reads the vault through that tool when it needs to.
"""
from __future__ import annotations

from src.agent.domain.entities.assistant import Assistant

_BLOCK_SEPARATOR = "\n\n"


class SystemPromptComposer:
    """Join the assistant prompt with the rendered context blocks."""

    def compose(
        self,
        assistant: Assistant,
        calendar_block: str,
        conversation_state_block: str,
        memory_pointer: str,
    ) -> str:
        """Return the system message text.

        The order is fixed: the assistant's prompt, the calendar block, the
        conversation-state block, the long-term-memory pointer. A blank block is
        left out together with its separator, so a chat without recorded state
        gets no empty section. Blocks are separated by one blank line.
        """
        blocks = (
            assistant.system_prompt,
            calendar_block,
            conversation_state_block,
            memory_pointer,
        )
        return _BLOCK_SEPARATOR.join(
            block.strip() for block in blocks if block and block.strip()
        )
