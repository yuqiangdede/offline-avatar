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
from packages.core import events
from packages.core.config import AppConfig, ensure_project_dir, resolve_project_path
from packages.core.interfaces import ASRProvider, AvatarProvider, LLMProvider, TTSProvider
from packages.providers.asr_faster_whisper import FasterWhisperProvider
from packages.providers.avatar_lite_avatar import LiteAvatarProvider
from packages.providers.llm_openai_compat import OpenAICompatLocalProvider
from packages.providers.tts_pyttsx3 import Pyttsx3Provider

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

    zh_rules = ["用中文回答", "请用中文回答", "用中文回复", "中文回答", "reply in chinese", "answer in chinese"]
    en_rules = ["用英文回答", "请用英文回答", "用英语回答", "英文回答", "reply in english", "answer in english"]

    if any(rule in lowered for rule in zh_rules):
        return "zh"
    if any(rule in lowered for rule in en_rules):
        return "en"
    return None


def _split_sentences(text: str) -> list[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[。！？；.!?])", text)
    cleaned = [p.strip() for p in parts if p and p.strip()]
    return cleaned if cleaned else [text.strip()]


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


def _prefer_single_avatar_render(avatar: AvatarProvider) -> bool:
    """
    Lite-Avatar CLI has high process startup cost. Rendering per sentence causes
    repeated model load and very high CPU time.
    """
    if not isinstance(avatar, LiteAvatarProvider):
        return False
    return bool((avatar.lite_avatar_cli or "").strip())


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
    raise ValueError(f"未知 ASR Provider: {name}")


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
    raise ValueError(f"未知 LLM Provider: {name}")


def _build_tts_provider(config: AppConfig) -> TTSProvider:
    name = config.providers.tts
    if name == "pyttsx3":
        return Pyttsx3Provider(
            rate=config.tts.rate,
            volume=config.tts.volume,
            temp_dir=ensure_project_dir(config, config.runtime.temp_dir),
        )
    raise ValueError(f"未知 TTS Provider: {name}")


def _build_avatar_provider(config: AppConfig) -> AvatarProvider:
    name = config.providers.avatar
    if name == "lite_avatar":
        return LiteAvatarProvider(
            fps=config.webrtc.fps,
            width=config.webrtc.width,
            height=config.webrtc.height,
            lite_avatar_cli=config.avatar.lite_avatar_cli,
            gpu_index=config.avatar.gpu_index,
            ffmpeg_path=resolve_project_path(config, config.avatar.ffmpeg_path)
            if (config.avatar.ffmpeg_path or "").strip()
            else "",
            temp_dir=ensure_project_dir(config, config.runtime.temp_dir),
            modelscope_cache_dir=ensure_project_dir(config, config.runtime.modelscope_cache_dir),
            workdir=str(config.project_root),
        )
    raise ValueError(f"未知 Avatar Provider: {name}")


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
            assistant_text = "本地 LLM 调用失败，请检查服务状态。" if reply_lang == "zh" else "Local LLM call failed."
        llm_ms = Metrics.elapsed_ms(llm_start)
        logger.info("Session[%s] LLM elapsed_ms=%s", self.session_id, llm_ms)

        if not assistant_text:
            assistant_text = "我没有获取到可用回复。" if reply_lang == "zh" else "I did not get a usable response."
            logger.warning("Session[%s] LLM returned empty text", self.session_id)

        await self._send({"type": events.WS_TYPE_LLM_FINAL, "text": assistant_text, "ms": llm_ms})
        await self._send_metric("llm_ms", llm_ms)

        await self._append_chat("assistant", assistant_text, reply_lang)
        self.llm_messages.append({"role": "assistant", "content": assistant_text})

        sentences = _split_sentences(assistant_text)
        if _prefer_single_avatar_render(self.avatar):
            sentences = [assistant_text]
            logger.info(
                "Session[%s] avatar render mode=single_pass (lite-avatar cli)",
                self.session_id,
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
            if not pcm_s16le:
                logger.warning("Session[%s] TTS empty pcm: sentence_idx=%s", self.session_id, idx)
                continue

            avatar_start = Metrics.now()
            audio_duration_s = len(pcm_s16le) / float(max(1, sample_rate) * 2)
            target_frames = max(1, int(round(audio_duration_s * max(1, self.config.webrtc.fps))))

            def _render_frames() -> list[av.VideoFrame]:
                rendered = list(
                    self.avatar.render(
                        pcm_s16le=pcm_s16le,
                        sample_rate=sample_rate,
                        avatar_asset=self.avatar_asset,
                    )
                )
                return _align_frames_to_audio(rendered, target_frames)

            try:
                frames = await asyncio.to_thread(_render_frames)
                if not frames:
                    logger.warning("Session[%s] Avatar empty frames: sentence_idx=%s", self.session_id, idx)
                    continue
                await self.transport.enqueue_av_segment(
                    pcm_s16le=pcm_s16le,
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
