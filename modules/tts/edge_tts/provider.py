# File: offline-avatar/modules/tts/edge_tts/provider.py
from __future__ import annotations

import audioop
import os
import tempfile
import threading
import wave
from pathlib import Path
from typing import Any

import pyttsx3

from modules.core.interfaces import TTSProvider


class Pyttsx3Provider(TTSProvider):
    def __init__(self, rate: int = 185, volume: float = 1.0, temp_dir: str | None = None):
        self.engine = pyttsx3.init()
        self.rate = rate
        self.volume = volume
        self._lock = threading.Lock()
        self.temp_dir = temp_dir
        if self.temp_dir:
            Path(self.temp_dir).mkdir(parents=True, exist_ok=True)

    def _choose_voice(self, lang: str) -> str | None:
        voices = self.engine.getProperty("voices")
        target_words = ["zh", "chinese", "mandarin"] if lang == "zh" else ["en", "english"]

        for voice in voices:
            token = f"{voice.id} {voice.name}"
            langs = getattr(voice, "languages", [])
            if langs:
                token += " " + " ".join(str(x) for x in langs)
            token = token.lower()
            if any(word in token for word in target_words):
                return voice.id
        return None

    @staticmethod
    def _read_wav_pcm_s16le(path: str) -> dict[str, Any]:
        with wave.open(path, "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())

        if sample_width != 2:
            frames = audioop.lin2lin(frames, sample_width, 2)
        if channels > 1:
            frames = audioop.tomono(frames, 2, 0.5, 0.5)

        return {"pcm_s16le": frames, "sample_rate": sample_rate}

    def synthesize(self, text: str, lang: str) -> dict[str, Any]:
        text = (text or "").strip()
        if not text:
            return {"pcm_s16le": b"", "sample_rate": 16000}

        tmp_path = ""
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False,
                dir=self.temp_dir,
            ) as tmp:
                tmp_path = tmp.name

            with self._lock:
                voice = self._choose_voice(lang)
                if voice:
                    self.engine.setProperty("voice", voice)
                self.engine.setProperty("rate", self.rate)
                self.engine.setProperty("volume", self.volume)
                self.engine.save_to_file(text, tmp_path)
                self.engine.runAndWait()

            return self._read_wav_pcm_s16le(tmp_path)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
