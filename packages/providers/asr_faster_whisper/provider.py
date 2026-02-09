# File: offline-avatar/packages/providers/asr_faster_whisper/provider.py
from __future__ import annotations

import logging
import re
import threading
from typing import Any

import numpy as np

from packages.core.interfaces import ASRProvider

logger = logging.getLogger(__name__)

try:
    from faster_whisper import WhisperModel
except Exception:  # pragma: no cover
    WhisperModel = None


def _normalize_lang(raw: str | None, text: str) -> str:
    if raw:
        lang = raw.lower()
        if lang.startswith("zh"):
            return "zh"
        if lang.startswith("en"):
            return "en"
    zh_count = len(re.findall(r"[\u4e00-\u9fff]", text or ""))
    return "zh" if zh_count > 0 else "en"


class FasterWhisperProvider(ASRProvider):
    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        device_index: int = 0,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.device_index = int(device_index)
        self._model = None
        self._lock = threading.Lock()

    def _get_model(self):
        if WhisperModel is None:
            raise RuntimeError("faster-whisper 未安装，请先 pip install faster-whisper")
        if self._model is None:
            with self._lock:
                if self._model is None:
                    kwargs: dict[str, Any] = {
                        "device": self.device,
                        "compute_type": self.compute_type,
                    }
                    if str(self.device).lower().startswith("cuda"):
                        kwargs["device_index"] = self.device_index
                    self._model = WhisperModel(self.model_size, **kwargs)
                    logger.info(
                        "FasterWhisper model loaded: model=%s device=%s device_index=%s compute_type=%s",
                        self.model_size,
                        self.device,
                        self.device_index if str(self.device).lower().startswith("cuda") else "-",
                        self.compute_type,
                    )
        return self._model

    def transcribe(self, audio_bytes: bytes, sample_rate: int) -> dict[str, Any]:
        if not audio_bytes:
            return {"text": "", "lang": "zh", "segments": []}

        model = self._get_model()
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        segments_iter, info = model.transcribe(
            audio,
            language=None,
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
        )

        segments = []
        texts: list[str] = []
        for segment in segments_iter:
            seg_text = (segment.text or "").strip()
            if not seg_text:
                continue
            texts.append(seg_text)
            segments.append(
                {
                    "start": float(getattr(segment, "start", 0.0)),
                    "end": float(getattr(segment, "end", 0.0)),
                    "text": seg_text,
                }
            )

        text = " ".join(texts).strip()
        lang = _normalize_lang(getattr(info, "language", None), text)
        return {"text": text, "lang": lang, "segments": segments}
