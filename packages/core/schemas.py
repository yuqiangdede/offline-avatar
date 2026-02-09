# File: offline-avatar/packages/core/schemas.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChatMessage:
    role: str
    text: str
    lang: str
    ts: int
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionMetric:
    name: str
    value: int
