# File: offline-avatar/packages/core/interfaces.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from av import VideoFrame


class ASRProvider(ABC):
    @abstractmethod
    def transcribe(self, audio_bytes: bytes, sample_rate: int) -> dict:
        """Return: {"text": str, "lang": "zh"|"en", "segments": optional}"""


class LLMProvider(ABC):
    @abstractmethod
    def chat(self, messages: list, lang: str, stream: bool = False):
        """Return iterator (stream=True) or str (stream=False)."""


class TTSProvider(ABC):
    @abstractmethod
    def synthesize(self, text: str, lang: str) -> dict:
        """Return: {"pcm_s16le": bytes, "sample_rate": int}"""


class AvatarProvider(ABC):
    @abstractmethod
    def render(
        self,
        pcm_s16le: bytes,
        sample_rate: int,
        avatar_asset: str,
    ) -> Iterator[VideoFrame]:
        """Yield av.VideoFrame for the avatar video stream."""
