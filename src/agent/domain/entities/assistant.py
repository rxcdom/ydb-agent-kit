from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Assistant:
    """Configuration of the single assistant: its prompt and generation settings.

    It is built from application settings at start-up, not loaded from a table.
    """

    name: str
    system_prompt: str
    model_name: str
    temperature: float

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Assistant name cannot be empty")
        if not self.system_prompt or not self.system_prompt.strip():
            raise ValueError("Assistant system prompt cannot be empty")
        if not self.model_name or not self.model_name.strip():
            raise ValueError("Assistant model name cannot be empty")
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError("Temperature must be between 0.0 and 2.0")
