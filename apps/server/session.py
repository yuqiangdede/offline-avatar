# File: offline-avatar/apps/server/session.py
from __future__ import annotations

import asyncio
import audioop
import base64
import io
import logging
import re
import tempfile
import time
import traceback
import uuid
import wave
from pathlib import Path
from typing import Any
import zipfile

import av
import numpy as np

from apps.server.metrics import Metrics
from apps.server.webrtc import WebRTCTransport
from modules.core import events
from modules.core.config import AppConfig, ensure_project_dir, resolve_project_path
from modules.core.interfaces import ASRProvider, AvatarProvider, LLMProvider, TTSProvider
from modules.asr.faster_whisper import FasterWhisperProvider
from modules.avatar.lite_avatar import LiteAvatarProvider
from modules.llm.openai_adapter import OpenAICompatLocalProvider
from modules.tts.edge_tts import Pyttsx3Provider

logger = logging.getLogger(__name__)


def _shorten(text: str, max_len: int = 120) -> str:
    text = (text or "").replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def _detect_lang(text: str) -> str:
    text = text or ""
    zh_count = len(re.findall(r"[\u4e00-\u9fff]", text))
    ascii_count = len(re.findall(r"[A-Za-z]", text))
    return "zh" if zh_count >= max(1, ascii_count) else "en"


def _override_lang_from_text(text: str) -> str | None:
    lowered = (text or "").lower()

    zh_rules = [
        "\u7528\u4e2d\u6587\u56de\u7b54",
        "\u8bf7\u7528\u4e2d\u6587\u56de\u7b54",
        "reply in chinese",
        "answer in chinese",
    ]
    en_rules = [
        "\u7528\u82f1\u6587\u56de\u7b54",
        "\u8bf7\u7528\u82f1\u6587\u56de\u7b54",
        "reply in english",
        "answer in english",
    ]

    if any(rule in lowered for rule in zh_rules):
        return "zh"
    if any(rule in lowered for rule in en_rules):
        return "en"
    return None


def _split_sentences(text: str) -> list[str]:
    if not text:
        return []
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    major_parts = re.split(r"(?<=[\u3002\uff01\uff1f\uff1b.!?])", normalized)
    chunks: list[str] = []
    max_chunk_chars = 48

    for major in major_parts:
        major = major.strip()
        if not major:
            continue
        if len(major) <= max_chunk_chars:
            chunks.append(major)
            continue

        # If one sentence is too long, split again by pause punctuation first.
        minor_parts = re.split(r"(?<=[\uff0c,\u3001\uff1a:])", major)
        for minor in minor_parts:
            minor = minor.strip()
            if not minor:
                continue
            if len(minor) <= max_chunk_chars:
                chunks.append(minor)
                continue
            # Hard split as a final fallback for long no-punctuation text.
            start = 0
            while start < len(minor):
                end = min(start + max_chunk_chars, len(minor))
                piece = minor[start:end].strip()
                if piece:
                    chunks.append(piece)
                start = end

    return chunks if chunks else [normalized]


def _merge_short_sentences(sentences: list[str], min_chars: int) -> list[str]:
    cleaned = [s.strip() for s in sentences if s and s.strip()]
    if min_chars <= 1 or len(cleaned) <= 1:
        return cleaned

    merged: list[str] = []
    idx = 0
    while idx < len(cleaned):
        cur = cleaned[idx]
        if len(cur) < min_chars and idx + 1 < len(cleaned):
            nxt = cleaned[idx + 1]
            # Keep space only for latin-word boundaries.
            if re.search(r"[A-Za-z0-9]$", cur) and re.match(r"^[A-Za-z0-9]", nxt):
                merged.append(f"{cur} {nxt}")
            else:
                merged.append(f"{cur}{nxt}")
            idx += 2
            continue
        merged.append(cur)
        idx += 1
    return merged

def _align_frames_to_audio(frames: list[av.VideoFrame], target_frames: int) -> list[av.VideoFrame]:
    target = max(1, int(target_frames))
    if not frames:
        return []
    if len(frames) == target:
        return frames
    if len(frames) > target:
        return frames[:target]
    last = frames[-1]
    return frames + [last] * (target - len(frames))


def _pcm_rms_s16le(pcm_s16le: bytes) -> int:
    if not pcm_s16le:
        return 0
    try:
        return int(audioop.rms(pcm_s16le, 2))
    except Exception:
        return 0


def _build_tone_pcm_s16le(duration_s: float, sample_rate: int, freq_hz: float = 220.0) -> bytes:
    duration = max(0.2, float(duration_s))
    sr = max(8000, int(sample_rate))
    t = np.arange(int(duration * sr), dtype=np.float32) / float(sr)
    # Short fade-in/out to avoid click noise.
    fade_len = max(1, int(0.02 * sr))
    env = np.ones_like(t)
    env[:fade_len] = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
    env[-fade_len:] = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
    wave_np = np.sin(2.0 * np.pi * float(freq_hz) * t) * env
    return (wave_np * 7000.0).astype(np.int16).tobytes()


def _pad_pcm_tail_silence_s16le(pcm_s16le: bytes, sample_rate: int, min_duration_s: float) -> bytes:
    if not pcm_s16le:
        return pcm_s16le
    sr = max(8000, int(sample_rate))
    target_samples = int(max(0.0, float(min_duration_s)) * sr)
    cur_samples = len(pcm_s16le) // 2
    if cur_samples >= target_samples:
        return pcm_s16le
    need_samples = target_samples - cur_samples
    return pcm_s16le + (b"\x00" * need_samples * 2)

def _trim_pcm_silence_s16le(
    pcm_s16le: bytes,
    sample_rate: int,
    frame_ms: int = 20,
    min_rms: int = 70,
    keep_lead_ms: int = 120,
    keep_tail_ms: int = 120,
    max_side_trim_ms: int = 280,
    min_keep_ratio: float = 0.85,
) -> bytes:
    if not pcm_s16le:
        return pcm_s16le
    sr = max(8000, int(sample_rate))
    frame_samples = max(1, int(sr * frame_ms / 1000))
    frame_bytes = frame_samples * 2
    total = len(pcm_s16le)
    if total <= frame_bytes * 2:
        return pcm_s16le

    rms_values: list[int] = []
    offsets: list[int] = []
    peak_rms = 0
    offset = 0
    while offset < total:
        chunk = pcm_s16le[offset : min(total, offset + frame_bytes)]
        rms = _pcm_rms_s16le(chunk)
        rms_values.append(rms)
        offsets.append(offset)
        if rms > peak_rms:
            peak_rms = rms
        offset += frame_bytes

    # Dynamic threshold: avoid over-trimming low-volume speech.
    threshold = max(min_rms, int(peak_rms * 0.08))
    first_idx = None
    last_idx = None
    for i, rms in enumerate(rms_values):
        if rms >= threshold:
            first_idx = i
            break
    for i in range(len(rms_values) - 1, -1, -1):
        if rms_values[i] >= threshold:
            last_idx = i
            break

    if first_idx is None or last_idx is None or last_idx < first_idx:
        return pcm_s16le

    keep_lead_bytes = max(0, int(sr * keep_lead_ms / 1000) * 2)
    keep_tail_bytes = max(0, int(sr * keep_tail_ms / 1000) * 2)

    start = max(0, offsets[first_idx] - keep_lead_bytes)
    end_base = offsets[last_idx] + frame_bytes
    end = min(total, end_base + keep_tail_bytes)
    if end <= start:
        return pcm_s16le

    max_side_trim_bytes = max(0, int(sr * max_side_trim_ms / 1000) * 2)
    removed_lead = start
    removed_tail = total - end
    if removed_lead > max_side_trim_bytes or removed_tail > max_side_trim_bytes:
        return pcm_s16le

    trimmed = pcm_s16le[start:end]
    min_keep_bytes = int(total * max(0.1, min(1.0, float(min_keep_ratio))))
    if len(trimmed) < min_keep_bytes:
        return pcm_s16le

    # Avoid aggressive trim on near-silent or too-short result.
    if len(trimmed) < max(frame_bytes * 2, int(0.25 * sr) * 2):
        return pcm_s16le
    return trimmed


def _estimate_voice_span_ms(
    pcm_s16le: bytes,
    sample_rate: int,
    frame_ms: int = 20,
    min_rms: int = 70,
) -> tuple[int, int]:
    if not pcm_s16le:
        return 0, 0
    sr = max(8000, int(sample_rate))
    frame_samples = max(1, int(sr * frame_ms / 1000))
    frame_bytes = frame_samples * 2
    total = len(pcm_s16le)
    if total <= frame_bytes:
        return 0, int(total / float(sr * 2) * 1000)

    first_idx = None
    last_idx = None
    idx = 0
    offset = 0
    while offset < total:
        chunk = pcm_s16le[offset : min(total, offset + frame_bytes)]
        rms = _pcm_rms_s16le(chunk)
        if rms >= min_rms:
            if first_idx is None:
                first_idx = idx
            last_idx = idx
        idx += 1
        offset += frame_bytes

    if first_idx is None or last_idx is None:
        return 0, 0
    start_ms = first_idx * frame_ms
    end_ms = (last_idx + 1) * frame_ms
    return start_ms, end_ms


def _safe_ts_ms() -> int:
    return int(time.time() * 1000)


def _decode_wav_to_pcm_s16le(audio_bytes: bytes, target_sample_rate: int) -> tuple[bytes, int]:
    with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        pcm = wf.readframes(wf.getnframes())

    if sample_width != 2:
        pcm = audioop.lin2lin(pcm, sample_width, 2)
    if channels > 1:
        pcm = audioop.tomono(pcm, 2, 0.5, 0.5)
    if sample_rate != target_sample_rate:
        pcm, _ = audioop.ratecv(pcm, 2, 1, sample_rate, target_sample_rate, None)
        sample_rate = target_sample_rate

    return pcm, sample_rate


def _decode_webm_opus_to_pcm_s16le(audio_bytes: bytes, target_sample_rate: int) -> tuple[bytes, int]:
    container = av.open(io.BytesIO(audio_bytes), mode="r", format="webm")
    chunks: list[bytes] = []

    try:
        resampler = av.AudioResampler(format="s16", layout="mono", rate=target_sample_rate)
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                arr = out.to_ndarray()
                if arr.ndim == 2:
                    arr = arr[0]
                chunks.append(arr.astype(np.int16).tobytes())
    finally:
        container.close()

    return b"".join(chunks), target_sample_rate


def _pick_avatar_idle_video(avatar_dir: Path) -> Path | None:
    candidates = [
        avatar_dir / "bg_video_silence.mp4",
        avatar_dir / "bg_video.mp4",
        avatar_dir / "bg_video_palindrome.mp4",
    ]
    for path in candidates:
        if path.exists():
            return path

    any_mp4 = sorted(avatar_dir.rglob("*.mp4"))
    return any_mp4[0] if any_mp4 else None


def _load_idle_frame_from_asset(avatar_asset: str, temp_dir: str | None = None):
    asset = Path(avatar_asset)
    if not asset.exists():
        return None

    def _decode_first_frame(video_path: Path):
        container = av.open(str(video_path))
        try:
            for frame in container.decode(video=0):
                return frame
        finally:
            container.close()
        return None

    if asset.suffix.lower() != ".zip":
        if asset.is_dir():
            video = _pick_avatar_idle_video(asset)
            if video:
                return _decode_first_frame(video)
        return None

    with tempfile.TemporaryDirectory(dir=temp_dir) as td:
        tmp_dir = Path(td)
        with zipfile.ZipFile(asset, "r") as zf:
            zf.extractall(tmp_dir)

        dirs = [p for p in tmp_dir.iterdir() if p.is_dir()]
        avatar_dir = dirs[0] if len(dirs) == 1 else tmp_dir
        video = _pick_avatar_idle_video(avatar_dir)
        if video:
            return _decode_first_frame(video)
    return None


def _build_asr_provider(config: AppConfig) -> ASRProvider:
    name = config.providers.asr
    if name == "faster_whisper_small":
        model_ref = config.asr.model_size
        # Keep built-in size aliases (e.g. "small"), but resolve local paths from project root.
        if any(token in model_ref for token in ("/", "\\")) or model_ref.startswith(".") or Path(model_ref).is_absolute():
            model_ref = resolve_project_path(config, model_ref)
        return FasterWhisperProvider(
            model_size=model_ref,
            device=config.asr.device,
            compute_type=config.asr.compute_type,
            device_index=config.asr.device_index,
        )
    raise ValueError(f"Unknown ASR Provider: {name}")


def _build_llm_provider(config: AppConfig) -> LLMProvider:
    name = config.providers.llm
    if name == "openai_compat_local":
        return OpenAICompatLocalProvider(
            endpoint=config.llm.endpoint,
            model=config.llm.model,
            timeout_s=config.llm.timeout_s,
            system_prompt_zh=config.llm.system_prompt_zh,
            system_prompt_en=config.llm.system_prompt_en,
        )
    raise ValueError(f"Unknown LLM Provider: {name}")


def _build_tts_provider(config: AppConfig) -> TTSProvider:
    name = config.providers.tts
    if name == "pyttsx3":
        return Pyttsx3Provider(
            rate=config.tts.rate,
            volume=config.tts.volume,
            temp_dir=ensure_project_dir(config, config.runtime.temp_dir),
        )
    raise ValueError(f"Unknown TTS Provider: {name}")


def _build_avatar_provider(config: AppConfig) -> AvatarProvider:
    name = config.providers.avatar
    if name == "lite_avatar":
        return LiteAvatarProvider(
            fps=config.webrtc.fps,
            width=config.webrtc.width,
            height=config.webrtc.height,
            lite_avatar_cli=config.avatar.lite_avatar_cli,
            gpu_index=config.avatar.gpu_index,
            delete_generated_mp4=bool(config.avatar.delete_generated_mp4),
            delete_generated_audio=bool(config.avatar.delete_generated_audio),
            ffmpeg_path=resolve_project_path(config, config.avatar.ffmpeg_path)
            if (config.avatar.ffmpeg_path or "").strip()
            else "",
            temp_dir=ensure_project_dir(config, config.runtime.temp_dir),
            modelscope_cache_dir=ensure_project_dir(config, config.runtime.modelscope_cache_dir),
            workdir=str(config.project_root),
        )
    raise ValueError(f"Unknown Avatar Provider: {name}")


class Session:
    def __init__(self, config: AppConfig, send_json):
        self.config = config
        self.send_json = send_json
        self.session_id = uuid.uuid4().hex[:8]
        self.runtime_temp_dir = ensure_project_dir(config, config.runtime.temp_dir)

        self.asr = _build_asr_provider(config)
        self.llm = _build_llm_provider(config)
        self.tts = _build_tts_provider(config)
        self.avatar = _build_avatar_provider(config)

        self.avatar_asset = resolve_project_path(config, config.avatar.default_asset)

        self.transport = WebRTCTransport(config=config, signal_sender=self._send)
        self.chat_history: list[dict[str, Any]] = []
        self.llm_messages: list[dict[str, str]] = []

        try:
            idle_frame = _load_idle_frame_from_asset(
                self.avatar_asset,
                temp_dir=self.runtime_temp_dir,
            )
            if idle_frame is not None:
                self.transport.set_idle_frame(idle_frame)
                logger.info("Session[%s] idle avatar frame loaded", self.session_id)
            else:
                logger.warning("Session[%s] idle avatar frame not found, use default idle canvas", self.session_id)
        except Exception:
            logger.exception("Session[%s] failed to load idle avatar frame", self.session_id)

        self._tasks: set[asyncio.Task] = set()
        self._send_lock = asyncio.Lock()
        self._pipeline_lock = asyncio.Lock()
        self._closed = False
        logger.info(
            "Session[%s] created: avatar_asset=%s, asr=%s, llm=%s, tts=%s, avatar=%s",
            self.session_id,
            self.avatar_asset,
            config.providers.asr,
            config.providers.llm,
            config.providers.tts,
            config.providers.avatar,
        )

    def submit_text(self, text: str) -> None:
        self._spawn(self._handle_text(text))

    def submit_audio(self, fmt: str, data_base64: str) -> None:
        self._spawn(self._handle_audio(fmt, data_base64))

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(self._run_safe(coro))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run_safe(self, coro) -> None:
        try:
            await coro
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Session[%s] pipeline crashed", self.session_id)
            if not self._closed:
                try:
                    await self._send_state(events.PHASE_IDLE)
                except Exception:
                    pass

    async def handle_offer(self, sdp: str, sdp_type: str) -> dict[str, str]:
        logger.info("Session[%s] handle_offer: sdp_type=%s", self.session_id, sdp_type)
        return await self.transport.create_answer(sdp=sdp, sdp_type=sdp_type)

    async def handle_ice(self, candidate: dict | None) -> None:
        logger.info("Session[%s] handle_ice: has_candidate=%s", self.session_id, bool(candidate))
        await self.transport.add_ice_candidate(candidate)

    async def clear_chat(self) -> None:
        self.chat_history.clear()
        self.llm_messages.clear()
        logger.info("Session[%s] chat cleared", self.session_id)
        await self._send_state(events.PHASE_IDLE)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        logger.info("Session[%s] closing: task_count=%s", self.session_id, len(self._tasks))

        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        await self.transport.close()
        logger.info("Session[%s] closed", self.session_id)

    async def _send(self, payload: dict[str, Any]) -> None:
        if self._closed:
            return
        async with self._send_lock:
            await self.send_json(payload)

    async def _send_state(self, phase: str) -> None:
        await self._send({"type": events.WS_TYPE_STATE, "phase": phase})

    async def _send_metric(self, name: str, value: int) -> None:
        await self._send({"type": events.WS_TYPE_METRIC, "name": name, "value": int(value)})

    async def _append_chat(self, role: str, text: str, lang: str) -> None:
        item = {"role": role, "text": text, "lang": lang, "ts": _safe_ts_ms()}
        self.chat_history.append(item)
        await self._send({"type": events.WS_TYPE_CHAT_APPEND, **item})

    async def _handle_text(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return

        logger.info("Session[%s] text input: len=%s content=%s", self.session_id, len(text), _shorten(text))
        async with self._pipeline_lock:
            e2e_start = Metrics.now()
            user_lang = _detect_lang(text)
            await self._run_pipeline(
                user_text=text,
                user_lang=user_lang,
                e2e_start=e2e_start,
                asr_ms=None,
                thinking_already_sent=False,
            )

    async def _handle_audio(self, fmt: str, data_base64: str) -> None:
        if not data_base64:
            return

        logger.info(
            "Session[%s] audio input: format=%s base64_len=%s",
            self.session_id,
            fmt,
            len(data_base64),
        )
        async with self._pipeline_lock:
            e2e_start = Metrics.now()
            await self._send_state(events.PHASE_THINKING)

            raw = base64.b64decode(data_base64)
            decode_format = (fmt or "").lower()
            logger.info(
                "Session[%s] audio decoded: format=%s bytes=%s",
                self.session_id,
                decode_format,
                len(raw),
            )

            asr_start = Metrics.now()
            if decode_format == "wav":
                pcm, sample_rate = await asyncio.to_thread(
                    _decode_wav_to_pcm_s16le,
                    raw,
                    self.config.audio.sample_rate,
                )
            else:
                pcm, sample_rate = await asyncio.to_thread(
                    _decode_webm_opus_to_pcm_s16le,
                    raw,
                    self.config.audio.sample_rate,
                )

            result = await asyncio.to_thread(self.asr.transcribe, pcm, sample_rate)
            asr_ms = Metrics.elapsed_ms(asr_start)

            user_text = (result.get("text") or "").strip()
            user_lang = result.get("lang") or _detect_lang(user_text)
            logger.info(
                "Session[%s] ASR done: ms=%s lang=%s text=%s",
                self.session_id,
                asr_ms,
                user_lang,
                _shorten(user_text),
            )

            await self._send(
                {
                    "type": events.WS_TYPE_ASR_FINAL,
                    "text": user_text,
                    "lang": user_lang,
                    "ms": asr_ms,
                }
            )
            await self._send_metric("asr_ms", asr_ms)

            if not user_text:
                logger.warning("Session[%s] ASR empty text", self.session_id)
                await self._send_state(events.PHASE_IDLE)
                return

            await self._run_pipeline(
                user_text=user_text,
                user_lang=user_lang,
                e2e_start=e2e_start,
                asr_ms=asr_ms,
                thinking_already_sent=True,
            )

    def _llm_window(self) -> list[dict[str, str]]:
        max_messages = 12
        return self.llm_messages[-max_messages:]

    async def _run_pipeline(
        self,
        user_text: str,
        user_lang: str,
        e2e_start: float,
        asr_ms: int | None,
        thinking_already_sent: bool,
    ) -> None:
        user_lang = user_lang if user_lang in ("zh", "en") else _detect_lang(user_text)
        reply_lang = _override_lang_from_text(user_text) or user_lang
        logger.info(
            "Session[%s] pipeline start: user_lang=%s reply_lang=%s text=%s",
            self.session_id,
            user_lang,
            reply_lang,
            _shorten(user_text),
        )

        await self._append_chat("user", user_text, user_lang)
        if not thinking_already_sent:
            await self._send_state(events.PHASE_THINKING)

        self.llm_messages.append({"role": "user", "content": user_text})

        llm_start = Metrics.now()
        try:
            logger.info(
                "Session[%s] LLM request: endpoint=%s model=%s messages=%s",
                self.session_id,
                self.config.llm.endpoint,
                self.config.llm.model,
                len(self._llm_window()),
            )
            llm_output = await asyncio.to_thread(self.llm.chat, self._llm_window(), reply_lang, False)
            assistant_text = (llm_output or "").strip()
            logger.info(
                "Session[%s] LLM success: text=%s",
                self.session_id,
                _shorten(assistant_text),
            )
        except Exception:
            logger.exception("Session[%s] LLM failed", self.session_id)
            assistant_text = "Local LLM call failed."
        llm_ms = Metrics.elapsed_ms(llm_start)
        logger.info("Session[%s] LLM elapsed_ms=%s", self.session_id, llm_ms)

        if not assistant_text:
            assistant_text = "I did not get a usable response."
            logger.warning("Session[%s] LLM returned empty text", self.session_id)

        await self._send({"type": events.WS_TYPE_LLM_FINAL, "text": assistant_text, "ms": llm_ms})
        await self._send_metric("llm_ms", llm_ms)
        self.llm_messages.append({"role": "assistant", "content": assistant_text})
        await self._append_chat("assistant", assistant_text, reply_lang)

        sentences = _split_sentences(assistant_text)
        if bool(self.config.avatar.single_pass_render):
            sentences = [assistant_text]
            logger.info(
                "Session[%s] avatar render mode=single_pass (config.avatar.single_pass_render=true)",
                self.session_id,
            )
        else:
            merge_chars = max(0, int(self.config.avatar.short_chunk_merge_chars))
            if merge_chars > 1:
                merged_sentences = _merge_short_sentences(sentences, min_chars=merge_chars)
                if len(merged_sentences) != len(sentences):
                    logger.info(
                        "Session[%s] sentence merge: before=%s after=%s min_chars=%s",
                        self.session_id,
                        len(sentences),
                        len(merged_sentences),
                        merge_chars,
                    )
                sentences = merged_sentences
            logger.info(
                "Session[%s] avatar render mode=segmented chunks=%s",
                self.session_id,
                len(sentences),
            )
        await self._send_state(events.PHASE_SPEAKING)

        first_tts_ms = None
        first_avatar_ms = None

        for idx, sentence in enumerate(sentences, start=1):
            tts_start = Metrics.now()
            try:
                tts_result = await asyncio.to_thread(self.tts.synthesize, sentence, reply_lang)
            except Exception:
                logger.exception(
                    "Session[%s] TTS failed: sentence_idx=%s text=%s",
                    self.session_id,
                    idx,
                    _shorten(sentence),
                )
                tts_result = {"pcm_s16le": b"", "sample_rate": self.config.audio.sample_rate}
            tts_ms = Metrics.elapsed_ms(tts_start)
            logger.info(
                "Session[%s] TTS done: sentence_idx=%s ms=%s text=%s",
                self.session_id,
                idx,
                tts_ms,
                _shorten(sentence),
            )

            if first_tts_ms is None:
                first_tts_ms = tts_ms
                await self._send_metric("tts_ms", tts_ms)

            pcm_s16le = tts_result.get("pcm_s16le", b"")
            sample_rate = int(tts_result.get("sample_rate", self.config.audio.sample_rate))
            tts_rms = _pcm_rms_s16le(pcm_s16le)
            if (not pcm_s16le) or tts_rms < 80:
                logger.warning(
                    "Session[%s] TTS empty/silent pcm: sentence_idx=%s bytes=%s rms=%s, apply fallback",
                    self.session_id,
                    idx,
                    len(pcm_s16le),
                    tts_rms,
                )
                try:
                    fallback_text = "Audio unavailable."
                    fb = await asyncio.to_thread(self.tts.synthesize, fallback_text, "en")
                    fb_pcm = fb.get("pcm_s16le", b"")
                    fb_rate = int(fb.get("sample_rate", self.config.audio.sample_rate))
                    fb_rms = _pcm_rms_s16le(fb_pcm)
                    if fb_pcm and fb_rms >= 80:
                        pcm_s16le = fb_pcm
                        sample_rate = fb_rate
                        logger.warning(
                            "Session[%s] TTS fallback voice used: sentence_idx=%s bytes=%s rms=%s",
                            self.session_id,
                            idx,
                            len(pcm_s16le),
                            fb_rms,
                        )
                    else:
                        raise RuntimeError("fallback voice is empty/silent")
                except Exception:
                    tone_s = max(0.8, min(3.0, len(sentence) * 0.08))
                    sample_rate = self.config.audio.sample_rate
                    pcm_s16le = _build_tone_pcm_s16le(tone_s, sample_rate)
                    logger.warning(
                        "Session[%s] TTS fallback tone used: sentence_idx=%s duration_s=%.2f",
                        self.session_id,
                        idx,
                        tone_s,
                    )

            avatar_start = Metrics.now()
            before_bytes = len(pcm_s16le)
            before_dur_s = before_bytes / float(max(1, sample_rate) * 2)
            if bool(self.config.avatar.trim_tts_silence):
                pcm_s16le = _trim_pcm_silence_s16le(pcm_s16le=pcm_s16le, sample_rate=sample_rate)
            after_bytes = len(pcm_s16le)
            after_dur_s = after_bytes / float(max(1, sample_rate) * 2)
            if after_bytes != before_bytes:
                logger.info(
                    "Session[%s] TTS silence-trim: sentence_idx=%s before_ms=%s after_ms=%s",
                    self.session_id,
                    idx,
                    int(before_dur_s * 1000),
                    int(after_dur_s * 1000),
                )
            logger.info(
                "Session[%s] TTS audio ready: sentence_idx=%s bytes=%s sample_rate=%s rms=%s",
                self.session_id,
                idx,
                len(pcm_s16le),
                sample_rate,
                _pcm_rms_s16le(pcm_s16le),
            )
            playback_pcm_s16le = pcm_s16le
            render_pcm_s16le = playback_pcm_s16le
            render_sample_rate = sample_rate
            if isinstance(self.avatar, LiteAvatarProvider):
                target_render_sr = max(8000, int(self.config.avatar.lite_avatar_render_sample_rate))
                if render_sample_rate != target_render_sr and len(render_pcm_s16le) > 0:
                    try:
                        render_pcm_s16le, _ = audioop.ratecv(
                            render_pcm_s16le,
                            2,
                            1,
                            render_sample_rate,
                            target_render_sr,
                            None,
                        )
                        logger.info(
                            "Session[%s] Lite-Avatar render resample: sentence_idx=%s from_sr=%s to_sr=%s bytes=%s",
                            self.session_id,
                            idx,
                            render_sample_rate,
                            target_render_sr,
                            len(render_pcm_s16le),
                        )
                        render_sample_rate = target_render_sr
                    except Exception:
                        logger.exception(
                            "Session[%s] Lite-Avatar render resample failed: sentence_idx=%s from_sr=%s to_sr=%s",
                            self.session_id,
                            idx,
                            render_sample_rate,
                            target_render_sr,
                        )
            audio_duration_s = len(playback_pcm_s16le) / float(max(1, sample_rate) * 2)
            voice_start_ms, voice_end_ms = _estimate_voice_span_ms(playback_pcm_s16le, sample_rate=sample_rate)
            logger.info(
                "Session[%s] Audio voice span: sentence_idx=%s total_ms=%s voice_start_ms=%s voice_end_ms=%s voice_dur_ms=%s",
                self.session_id,
                idx,
                int(audio_duration_s * 1000),
                voice_start_ms,
                voice_end_ms,
                max(0, voice_end_ms - voice_start_ms),
            )
            target_frames = max(1, int(round(audio_duration_s * max(1, self.config.webrtc.fps))))
            if isinstance(self.avatar, LiteAvatarProvider) and bool((self.avatar.lite_avatar_cli or "").strip()):
                min_audio_ms = max(0, int(self.config.avatar.lite_avatar_min_audio_ms))
                cur_audio_ms = int(len(render_pcm_s16le) / float(max(1, render_sample_rate) * 2) * 1000)
                if min_audio_ms > 0 and cur_audio_ms < min_audio_ms:
                    render_pcm_s16le = _pad_pcm_tail_silence_s16le(
                        render_pcm_s16le,
                        sample_rate=render_sample_rate,
                        min_duration_s=float(min_audio_ms) / 1000.0,
                    )
                    logger.info(
                        "Session[%s] Lite-Avatar short-audio pad: sentence_idx=%s before_ms=%s target_ms=%s",
                        self.session_id,
                        idx,
                        cur_audio_ms,
                        min_audio_ms,
                    )

            def _render_frames() -> tuple[list[av.VideoFrame], int]:
                rendered = list(
                    self.avatar.render(
                        pcm_s16le=render_pcm_s16le,
                        sample_rate=render_sample_rate,
                        avatar_asset=self.avatar_asset,
                    )
                )
                raw_count = len(rendered)
                aligned = _align_frames_to_audio(rendered, target_frames)
                return aligned, raw_count

            try:
                frames, raw_frame_count = await asyncio.to_thread(_render_frames)
                if not frames:
                    logger.warning("Session[%s] Avatar empty frames: sentence_idx=%s", self.session_id, idx)
                    continue
                fps = max(1, int(self.config.webrtc.fps))
                playback_audio_ms = int(audio_duration_s * 1000)
                render_audio_ms = int(len(render_pcm_s16le) / float(max(1, render_sample_rate) * 2) * 1000)
                raw_video_ms = int(raw_frame_count * 1000 / fps)
                out_video_ms = int(len(frames) * 1000 / fps)
                logger.info(
                    "Session[%s] AV duration compare: sentence_idx=%s audio_playback_ms=%s audio_render_ms=%s render_sr=%s video_raw_ms=%s video_out_ms=%s delta_out_ms=%s frames_raw=%s frames_out=%s fps=%s",
                    self.session_id,
                    idx,
                    playback_audio_ms,
                    render_audio_ms,
                    render_sample_rate,
                    raw_video_ms,
                    out_video_ms,
                    out_video_ms - playback_audio_ms,
                    raw_frame_count,
                    len(frames),
                    fps,
                )
                await self.transport.enqueue_av_segment(
                    pcm_s16le=playback_pcm_s16le,
                    sample_rate=sample_rate,
                    frames=frames,
                )
                logger.info(
                    "Session[%s] Avatar enqueued synced segment: sentence_idx=%s frames=%s audio_ms=%s",
                    self.session_id,
                    idx,
                    len(frames),
                    int(audio_duration_s * 1000),
                )
            except Exception:
                logger.exception("Session[%s] Avatar render failed: sentence_idx=%s", self.session_id, idx)
            avatar_ms = Metrics.elapsed_ms(avatar_start)
            logger.info("Session[%s] Avatar done: sentence_idx=%s ms=%s", self.session_id, idx, avatar_ms)
            if first_avatar_ms is None:
                first_avatar_ms = avatar_ms
                await self._send_metric("avatar_ms", avatar_ms)

        e2e_ms = Metrics.elapsed_ms(e2e_start)
        await self._send_metric("e2e_ms", e2e_ms)
        logger.info("Session[%s] pipeline done: e2e_ms=%s", self.session_id, e2e_ms)

        await self._send_state(events.PHASE_IDLE)

    async def debug_dump_error(self, exc: Exception) -> None:
        await self._send(
            {
                "type": "error",
                "message": str(exc),
                "trace": traceback.format_exc(),
            }
        )

