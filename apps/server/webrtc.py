# File: offline-avatar/apps/server/webrtc.py
from __future__ import annotations

import asyncio
import audioop
import logging
import time
from fractions import Fraction
from typing import Any, Awaitable, Callable

import av
import aioice.ice as aioice_ice
import numpy as np
from aiortc import (
    RTCIceCandidate,
    RTCPeerConnection,
    RTCRtpSender,
    RTCSessionDescription,
    MediaStreamTrack,
    VideoStreamTrack,
)
from aiortc.mediastreams import MediaStreamError
from aiortc.sdp import candidate_from_sdp, candidate_to_sdp
from av import VideoFrame

from packages.core.config import AppConfig

SignalSender = Callable[[dict[str, Any]], Awaitable[None]]
logger = logging.getLogger(__name__)


class QueueVideoTrack(VideoStreamTrack):
    def __init__(self, width: int = 1280, height: int = 720, fps: int = 20):
        super().__init__()
        self.width = width
        self.height = height
        self.fps = fps
        # Keep enough buffered frames for long utterances to avoid frame drops.
        self.queue: asyncio.Queue[VideoFrame] = asyncio.Queue(maxsize=max(400, fps * 60))
        self._idle = self._build_idle_frame()

        self._clock_rate = 90000
        self._frame_duration = int(self._clock_rate / max(1, fps))
        self._timestamp = 0
        self._started_at: float | None = None
        self._sent_count = 0

    def _build_idle_frame(self) -> VideoFrame:
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas[:, :, 0] = 16
        canvas[:, :, 1] = 20
        canvas[:, :, 2] = 28
        frame = VideoFrame.from_ndarray(canvas, format="rgb24")
        return frame

    def push(self, frame: VideoFrame) -> None:
        if self.readyState != "live":
            return
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            self.queue.put_nowait(frame)
        except asyncio.QueueFull:
            pass

    def push_many(self, frames: list[VideoFrame]) -> None:
        for frame in frames:
            self.push(frame)

    def set_idle_frame(self, frame: VideoFrame) -> None:
        if self.readyState != "live":
            return
        self._idle = frame

    def _fit_to_canvas(self, frame: VideoFrame) -> VideoFrame:
        src_w = int(getattr(frame, "width", 0) or 0)
        src_h = int(getattr(frame, "height", 0) or 0)
        if src_w <= 0 or src_h <= 0:
            return self._idle.reformat(width=self.width, height=self.height, format="yuv420p")

        if src_w == self.width and src_h == self.height:
            return frame.reformat(width=self.width, height=self.height, format="yuv420p")

        scale = min(self.width / src_w, self.height / src_h)
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))

        resized = frame.reformat(width=new_w, height=new_h, format="rgb24")
        arr = resized.to_ndarray()
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        top = (self.height - new_h) // 2
        left = (self.width - new_w) // 2
        canvas[top : top + new_h, left : left + new_w, :] = arr

        out = VideoFrame.from_ndarray(canvas, format="rgb24")
        return out.reformat(width=self.width, height=self.height, format="yuv420p")

    async def recv(self) -> VideoFrame:
        if self.readyState != "live":
            raise MediaStreamError

        if self._started_at is None:
            self._started_at = time.time()
            self._timestamp = 0
        else:
            self._timestamp += self._frame_duration

        target = self._started_at + (self._timestamp / self._clock_rate)
        delay = target - time.time()
        if delay > 0:
            await asyncio.sleep(delay)

        try:
            frame = self.queue.get_nowait()
            used_idle = False
        except asyncio.QueueEmpty:
            frame = self._idle
            used_idle = True

        out = self._fit_to_canvas(frame)
        out.pts = self._timestamp
        out.time_base = Fraction(1, self._clock_rate)
        self._sent_count += 1
        if self._sent_count == 1 or self._sent_count % 200 == 0:
            logger.info(
                "WebRTC video recv: sent=%s queue=%s idle=%s",
                self._sent_count,
                self.queue.qsize(),
                used_idle,
            )
        return out


class QueueAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, sample_rate: int = 16000, frame_ms: int = 20):
        super().__init__()
        self.sample_rate = sample_rate
        self.frame_samples = max(1, int(sample_rate * frame_ms / 1000))
        self.frame_bytes = self.frame_samples * 2
        # 60s buffer by default (frame_ms=20 -> 3000 chunks).
        self.queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=max(3000, int(60000 / frame_ms)))
        self._timestamp = 0
        self._sent_count = 0

    async def enqueue_pcm(self, pcm_s16le: bytes, sample_rate: int) -> None:
        if not pcm_s16le:
            return
        if sample_rate != self.sample_rate:
            pcm_s16le, _ = audioop.ratecv(pcm_s16le, 2, 1, sample_rate, self.sample_rate, None)

        offset = 0
        total = len(pcm_s16le)
        while offset < total:
            chunk = pcm_s16le[offset : offset + self.frame_bytes]
            offset += self.frame_bytes
            if len(chunk) < self.frame_bytes:
                chunk += b"\x00" * (self.frame_bytes - len(chunk))
            await self.queue.put(chunk)

    async def recv(self) -> av.AudioFrame:
        if self.readyState != "live":
            raise MediaStreamError

        try:
            chunk = await asyncio.wait_for(self.queue.get(), timeout=0.02)
            used_silence = False
        except asyncio.TimeoutError:
            chunk = b"\x00" * self.frame_bytes
            used_silence = True

        if len(chunk) > self.frame_bytes:
            chunk = chunk[: self.frame_bytes]
        elif len(chunk) < self.frame_bytes:
            chunk += b"\x00" * (self.frame_bytes - len(chunk))

        pcm = np.frombuffer(chunk, dtype=np.int16).reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(pcm, format="s16", layout="mono")
        frame.sample_rate = self.sample_rate
        frame.pts = self._timestamp
        frame.time_base = Fraction(1, self.sample_rate)
        self._timestamp += frame.samples
        self._sent_count += 1
        if self._sent_count == 1 or self._sent_count % 250 == 0:
            logger.info(
                "WebRTC audio recv: sent=%s queue=%s silence=%s",
                self._sent_count,
                self.queue.qsize(),
                used_silence,
            )
        return frame


class WebRTCTransport:
    def __init__(self, config: AppConfig, signal_sender: SignalSender):
        self.config = config
        self.signal_sender = signal_sender
        self.pc: RTCPeerConnection | None = None

        self.video_track = self._new_video_track()
        self.audio_track = self._new_audio_track()

    def _new_video_track(self) -> QueueVideoTrack:
        return QueueVideoTrack(
            width=self.config.webrtc.width,
            height=self.config.webrtc.height,
            fps=self.config.webrtc.fps,
        )

    def _new_audio_track(self) -> QueueAudioTrack:
        return QueueAudioTrack(sample_rate=self.config.audio.sample_rate)

    async def create_answer(self, sdp: str, sdp_type: str) -> dict[str, str]:
        previous_video = self.video_track
        previous_audio = self.audio_track
        await self.close()
        # Recreate media tracks per peer connection to avoid queue reuse across
        # reconnects (which can truncate audio head/tail).
        self.video_track = self._new_video_track()
        self.audio_track = self._new_audio_track()
        if previous_video is not None:
            try:
                self.video_track.set_idle_frame(previous_video._idle)
            except Exception:
                pass
            try:
                previous_video.stop()
            except Exception:
                pass
        if previous_audio is not None:
            try:
                previous_audio.stop()
            except Exception:
                pass
        self.pc = RTCPeerConnection()
        # Apply consent config defensively per connection to avoid stale defaults.
        consent_interval = max(1, int(self.config.webrtc.consent_interval_s))
        consent_failures = max(3, int(self.config.webrtc.consent_failures))
        aioice_ice.CONSENT_INTERVAL = consent_interval
        aioice_ice.CONSENT_FAILURES = consent_failures
        logger.info(
            "WebRTC consent effective: interval_s=%s failures=%s timeout_est_single=%ss timeout_est_dual=%ss",
            aioice_ice.CONSENT_INTERVAL,
            aioice_ice.CONSENT_FAILURES,
            aioice_ice.CONSENT_INTERVAL * aioice_ice.CONSENT_FAILURES,
            aioice_ice.CONSENT_INTERVAL * max(1, aioice_ice.CONSENT_FAILURES // 2),
        )

        @self.pc.on("icecandidate")
        async def on_icecandidate(candidate):
            if candidate is None:
                await self.signal_sender({"type": "webrtc.ice", "candidate": None})
                return
            await self.signal_sender(
                {
                    "type": "webrtc.ice",
                    "candidate": {
                        "candidate": "candidate:" + candidate_to_sdp(candidate),
                        "sdpMid": candidate.sdpMid,
                        "sdpMLineIndex": candidate.sdpMLineIndex,
                    },
                }
            )

        @self.pc.on("iceconnectionstatechange")
        async def on_ice_connection_state_change():
            logger.info("WebRTC iceConnectionState=%s", self.pc.iceConnectionState if self.pc else "closed")

        @self.pc.on("connectionstatechange")
        async def on_connection_state_change():
            logger.info("WebRTC connectionState=%s", self.pc.connectionState if self.pc else "closed")

        self.pc.addTrack(self.video_track)
        self.pc.addTrack(self.audio_track)

        self._apply_video_codec_preference(self.config.webrtc.video_codec)

        await self.pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=sdp_type))
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)

        local = self.pc.localDescription
        await self._prime_media()
        answer_sdp = self._filter_local_candidates_in_sdp(local.sdp)
        return {"sdp": answer_sdp, "sdpType": local.type}

    def _apply_video_codec_preference(self, codec_name: str) -> None:
        if not self.pc:
            return
        target = f"video/{codec_name.lower()}"

        capabilities = RTCRtpSender.getCapabilities("video")
        preferred = [c for c in capabilities.codecs if c.mimeType.lower() == target]
        if not preferred:
            return

        for transceiver in self.pc.getTransceivers():
            if transceiver.kind == "video":
                transceiver.setCodecPreferences(preferred)

    def _filter_local_candidates_in_sdp(self, sdp: str) -> str:
        force_ip = (self.config.webrtc.force_interface_ip or "").strip()
        if not force_ip:
            return sdp

        def _candidate_ip(line: str) -> str:
            # RFC5245 candidate fields:
            # foundation component transport priority ip port typ ...
            payload = line[len("a=candidate:") :]
            parts = payload.split()
            if len(parts) >= 6:
                return parts[4]
            return ""

        raw_lines = sdp.splitlines()
        kept_lines: list[str] = []
        total_candidates = 0
        kept_candidates = 0
        for line in raw_lines:
            normalized = line.rstrip("\r")
            if normalized.startswith("a=candidate:"):
                total_candidates += 1
                ip = _candidate_ip(normalized)
                if ip != force_ip:
                    continue
                kept_candidates += 1
            kept_lines.append(normalized)

        if total_candidates > 0 and kept_candidates == 0:
            logger.warning(
                "WebRTC force_interface_ip=%s no candidate matched, keep original SDP candidates",
                force_ip,
            )
            return sdp

        logger.info(
            "WebRTC force_interface_ip=%s candidate kept=%s/%s",
            force_ip,
            kept_candidates,
            total_candidates,
        )
        return "\r\n".join(kept_lines) + "\r\n"

    def _is_supported_remote_candidate(self, candidate: RTCIceCandidate) -> bool:
        ip = (candidate.ip or "").strip().lower()
        protocol = (candidate.protocol or "").strip().lower()
        force_ip = (self.config.webrtc.force_interface_ip or "").strip().lower()
        if not ip:
            return False
        if protocol and protocol != "udp":
            return False
        if ip.endswith(".local"):
            # mDNS hostname from browser host candidate, keep it.
            return True
        if force_ip and ip != force_ip:
            return False
        if ":" in ip:
            # Prefer IPv4 on Windows local deployment to avoid unstable
            # multi-NIC IPv6 candidate switching.
            return False
        return True

    async def add_ice_candidate(self, candidate_data: dict[str, Any] | None) -> None:
        if not self.pc:
            return
        if candidate_data is None:
            # aiortc 1.9.0 does not support addIceCandidate(None).
            return

        raw = candidate_data.get("candidate")
        if not raw:
            # End-of-candidates marker, safe to ignore for current aiortc.
            return

        raw_sdp = raw[10:] if raw.startswith("candidate:") else raw
        ice = candidate_from_sdp(raw_sdp)
        ice.sdpMid = candidate_data.get("sdpMid")
        ice.sdpMLineIndex = candidate_data.get("sdpMLineIndex")

        if isinstance(ice, RTCIceCandidate):
            if not self._is_supported_remote_candidate(ice):
                logger.info(
                    "WebRTC skip remote candidate: ip=%s protocol=%s type=%s",
                    ice.ip,
                    ice.protocol,
                    ice.type,
                )
                return
            await self.pc.addIceCandidate(ice)

    async def enqueue_audio(self, pcm_s16le: bytes, sample_rate: int) -> None:
        await self.audio_track.enqueue_pcm(pcm_s16le=pcm_s16le, sample_rate=sample_rate)

    async def enqueue_av_segment(
        self,
        pcm_s16le: bytes,
        sample_rate: int,
        frames: list[VideoFrame],
    ) -> None:
        # Queue audio/video together for the same utterance to improve A/V sync.
        self.video_track.push_many(frames)
        await self.audio_track.enqueue_pcm(pcm_s16le=pcm_s16le, sample_rate=sample_rate)

    def set_idle_frame(self, frame: VideoFrame) -> None:
        self.video_track.set_idle_frame(frame)

    def push_video_frame(self, frame: VideoFrame) -> None:
        self.video_track.push(frame)

    async def _prime_media(self) -> None:
        # Kick-start RTP flow with a short idle segment so browser can receive
        # audio/video packets before heavy avatar rendering finishes.
        warm_video_frames = max(1, int(self.video_track.fps * 2))
        self.video_track.push_many([self.video_track._idle] * warm_video_frames)

        warm_audio_samples = self.audio_track.sample_rate * 2
        warm_silence = b"\x00" * warm_audio_samples * 2
        await self.audio_track.enqueue_pcm(
            pcm_s16le=warm_silence,
            sample_rate=self.audio_track.sample_rate,
        )
        logger.info(
            "WebRTC media primed: video_frames=%s audio_ms=%s",
            warm_video_frames,
            2000,
        )

    async def close(self) -> None:
        if self.pc:
            await self.pc.close()
            self.pc = None
