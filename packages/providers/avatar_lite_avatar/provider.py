# File: offline-avatar/packages/providers/avatar_lite_avatar/provider.py
from __future__ import annotations

import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Iterator
import zipfile

import av
import numpy as np
from av import VideoFrame

from packages.core.interfaces import AvatarProvider

logger = logging.getLogger(__name__)


class LiteAvatarProvider(AvatarProvider):
    def __init__(
        self,
        fps: int = 20,
        width: int = 1280,
        height: int = 720,
        lite_avatar_cli: str = "",
        gpu_index: int = 0,
        ffmpeg_path: str = "",
        temp_dir: str | None = None,
        modelscope_cache_dir: str | None = None,
        workdir: str | None = None,
    ):
        self.fps = fps
        self.width = width
        self.height = height
        self.lite_avatar_cli = (lite_avatar_cli or "").strip()
        self.gpu_index = int(gpu_index)
        self.ffmpeg_path = (ffmpeg_path or "").strip()
        self.temp_dir = (temp_dir or "").strip() or None
        self.modelscope_cache_dir = (modelscope_cache_dir or "").strip() or None
        self.workdir = workdir
        self._cli_disabled_reason: str | None = None
        if self.temp_dir:
            Path(self.temp_dir).mkdir(parents=True, exist_ok=True)
        if self.modelscope_cache_dir:
            Path(self.modelscope_cache_dir).mkdir(parents=True, exist_ok=True)

    def _resolve_cli_cwd(self, cmd_parts: list[str]) -> str | None:
        """
        Prefer the directory of lite_avatar.py so upstream relative paths
        (e.g. ./weights/...) work on Windows.
        """
        for raw in cmd_parts[1:]:
            part = (raw or "").strip().strip('"').strip("'")
            if not part:
                continue
            path = Path(part)
            if not path.is_absolute() and self.workdir:
                path = Path(self.workdir) / path
            try:
                resolved = path.resolve()
            except Exception:
                continue
            if resolved.exists() and resolved.suffix.lower() == ".py":
                return str(resolved.parent)
        return self.workdir or None

    def _normalize_python_script_arg(self, cmd_parts: list[str]) -> list[str]:
        """
        Normalize python script arg to absolute path so it won't be duplicated by cwd.
        Example:
          cwd = .../models/lite-avatar
          cmd = python models/lite-avatar/lite_avatar.py
        """
        if len(cmd_parts) < 2:
            return cmd_parts

        script_idx = None
        for idx in range(1, len(cmd_parts)):
            raw = (cmd_parts[idx] or "").strip().strip('"').strip("'")
            if not raw or raw.startswith("-"):
                continue
            if Path(raw).suffix.lower() == ".py":
                script_idx = idx
                break

        if script_idx is None:
            return cmd_parts

        script_raw = (cmd_parts[script_idx] or "").strip().strip('"').strip("'")
        script_path = Path(script_raw)
        if script_path.is_absolute():
            return cmd_parts

        base = Path(self.workdir) if self.workdir else Path.cwd()
        resolved = (base / script_path).resolve()
        if not resolved.exists():
            return cmd_parts

        normalized = list(cmd_parts)
        normalized[script_idx] = str(resolved)
        return normalized

    def _ensure_cli_runtime_args(self, cmd_parts: list[str]) -> list[str]:
        """
        Ensure CLI gets runtime args consistent with server settings.
        """
        if not cmd_parts:
            return cmd_parts

        normalized = list(cmd_parts)
        lower_parts = [p.lower() for p in normalized]
        if "--fps" not in lower_parts:
            normalized += ["--fps", str(max(1, int(self.fps)))]
        if "--num_threads" not in lower_parts:
            normalized += ["--num_threads", "2"]
        if "--gpu_index" not in lower_parts:
            normalized += ["--gpu_index", str(self.gpu_index)]
        if self.ffmpeg_path and "--ffmpeg_path" not in lower_parts:
            normalized += ["--ffmpeg_path", self.ffmpeg_path]
        return normalized

    @staticmethod
    def _normalize_python_executable(cmd_parts: list[str]) -> list[str]:
        """
        If CLI starts with a generic Python launcher command, use current interpreter
        to avoid package mismatch between parent process and subprocess.
        """
        if not cmd_parts:
            return cmd_parts

        exe = (cmd_parts[0] or "").strip().strip('"').strip("'").lower()
        if exe in {"python", "python3", "python.exe", "python3.exe", "py", "py.exe"}:
            normalized = list(cmd_parts)
            normalized[0] = sys.executable
            return normalized
        return cmd_parts

    def _write_wav(self, path: Path, pcm_s16le: bytes, sample_rate: int) -> None:
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_s16le)

    def _resolve_ffmpeg_bin(self) -> str | None:
        explicit = (self.ffmpeg_path or "").strip().strip('"').strip("'")
        if explicit and Path(explicit).exists():
            return explicit

        env_path = (Path((self.workdir or Path.cwd())) if self.workdir else Path.cwd())
        candidates = []
        candidates.extend(env_path.glob("models/ffmpeg*/bin/ffmpeg.exe"))
        candidates.extend(env_path.glob("ffmpeg*/bin/ffmpeg.exe"))
        for c in candidates:
            if c.exists():
                return str(c.resolve())

        ffmpeg_env = (shutil.which("ffmpeg") or shutil.which("ffmpeg.exe"))
        return ffmpeg_env

    def _try_mux_mp4_from_frames(self, frame_dir: Path, wav_path: Path, result_dir: Path) -> Path | None:
        ffmpeg_bin = self._resolve_ffmpeg_bin()
        if not ffmpeg_bin:
            logger.warning("ffmpeg not found in PATH, skip post-mux from tmp_frames")
            return None

        out_path = result_dir / "test_demo_muxed.mp4"
        cmd_nvenc = [
            ffmpeg_bin,
            "-r",
            "30",
            "-i",
            str(frame_dir / "%05d.jpg"),
            "-i",
            str(wav_path),
            "-framerate",
            "30",
            "-c:v",
            "h264_nvenc",
            "-gpu",
            str(self.gpu_index),
            "-pix_fmt",
            "yuv420p",
            "-b:v",
            "5000k",
            "-loglevel",
            "error",
            str(out_path),
            "-y",
        ]
        proc = subprocess.run(cmd_nvenc, capture_output=True, text=True)
        if proc.returncode != 0:
            cmd_cpu = [
                ffmpeg_bin,
                "-r",
                "30",
                "-i",
                str(frame_dir / "%05d.jpg"),
                "-i",
                str(wav_path),
                "-framerate",
                "30",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-b:v",
                "5000k",
                "-strict",
                "experimental",
                "-loglevel",
                "error",
                str(out_path),
                "-y",
            ]
            proc = subprocess.run(cmd_cpu, capture_output=True, text=True)
            if proc.returncode == 0:
                logger.info("Lite-Avatar post-mux fallback to cpu encoder: libx264")
        if proc.returncode != 0:
            logger.warning(
                "Lite-Avatar post-mux failed(returncode=%s): stderr=%s",
                proc.returncode,
                (proc.stderr or "").strip()[:1000],
            )
            return None
        if not out_path.exists() or out_path.stat().st_size <= 0:
            logger.warning("Lite-Avatar post-mux produced empty mp4: %s", out_path)
            return None

        logger.info("Lite-Avatar post-mux output video=%s", out_path)
        return out_path

    @staticmethod
    def _pingpong_idx(frame_index: int, frame_count: int) -> int:
        if frame_count <= 1:
            return 0
        period = frame_count * 2 - 2
        mod = frame_index % period
        return mod if mod < frame_count else period - mod

    def _render_with_lite_avatar_cli(
        self,
        pcm_s16le: bytes,
        sample_rate: int,
        avatar_asset: str,
    ) -> Iterator[VideoFrame]:
        with tempfile.TemporaryDirectory(dir=self.temp_dir) as td:
            tmp_dir = Path(td)
            wav_path = tmp_dir / "input.wav"
            result_dir = tmp_dir / "result"
            result_dir.mkdir(parents=True, exist_ok=True)
            self._write_wav(wav_path, pcm_s16le, sample_rate)

            data_dir = Path(avatar_asset)
            if data_dir.suffix.lower() == ".zip":
                extracted_dir = tmp_dir / "avatar_data"
                extracted_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(data_dir, "r") as zf:
                    zf.extractall(extracted_dir)
                # If zip contains a single top-level directory, use it as data_dir.
                dirs = [p for p in extracted_dir.iterdir() if p.is_dir()]
                data_dir = dirs[0] if len(dirs) == 1 else extracted_dir

            cmd = shlex.split(self.lite_avatar_cli) + [
                "--data_dir",
                str(data_dir),
                "--audio_file",
                str(wav_path),
                "--result_dir",
                str(result_dir),
            ]
            cmd = self._ensure_cli_runtime_args(cmd)
            cmd = self._normalize_python_executable(cmd)
            cmd = self._normalize_python_script_arg(cmd)
            run_cwd = self._resolve_cli_cwd(cmd)
            logger.info("Lite-Avatar CLI start: cmd=%s cwd=%s", " ".join(cmd), run_cwd or ".")
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_index)
            if self.temp_dir:
                env["TMP"] = self.temp_dir
                env["TEMP"] = self.temp_dir
                env["TMPDIR"] = self.temp_dir
            if self.modelscope_cache_dir:
                env["MODELSCOPE_CACHE"] = self.modelscope_cache_dir
            if self.ffmpeg_path:
                env["FFMPEG_PATH"] = self.ffmpeg_path
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=run_cwd, env=env)
            if proc.returncode != 0:
                stderr_text = (proc.stderr or "").strip()
                missing = re.search(r"No module named '([^']+)'", stderr_text)
                if missing:
                    logger.error(
                        "Lite-Avatar 缺少 Python 依赖: %s。请安装后重启服务。",
                        missing.group(1),
                    )
                raise RuntimeError(
                    f"Lite-Avatar CLI 失败(returncode={proc.returncode}): "
                    f"stderr={stderr_text[:4000]} stdout={(proc.stdout or '').strip()[:1000]}"
                )

            stdout_text = (proc.stdout or "").strip()
            stderr_text = (proc.stderr or "").strip()
            key_lines: list[str] = []
            for line in (stdout_text + "\n" + stderr_text).splitlines():
                lower = line.lower()
                if (
                    "audio2mouth provider" in lower
                    or "liteavatar device" in lower
                    or "cuda unavailable" in lower
                    or "cudaexecutionprovider" in lower
                ):
                    key_lines.append(line.strip())
            if key_lines:
                logger.info("Lite-Avatar CLI runtime: %s", " | ".join(key_lines[:6]))

            videos = sorted(result_dir.rglob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
            if videos:
                out_path = videos[0]
                logger.info("Lite-Avatar CLI output video=%s", out_path)

                container = av.open(str(out_path))
                try:
                    for frame in container.decode(video=0):
                        yield frame
                finally:
                    container.close()
                return

            # If upstream didn't produce mp4, try muxing from tmp_frames here first.
            frame_dir = result_dir / "tmp_frames"
            if frame_dir.exists():
                muxed = self._try_mux_mp4_from_frames(frame_dir, wav_path, result_dir)
                if muxed:
                    container = av.open(str(muxed))
                    try:
                        for frame in container.decode(video=0):
                            yield frame
                    finally:
                        container.close()
                    return

            # Final fallback: stream generated jpg frames directly.
            jpgs = sorted(frame_dir.glob("*.jpg"))
            if not jpgs:
                raise RuntimeError("Lite-Avatar CLI 未生成 mp4 或 tmp_frames jpg")
            logger.info("Lite-Avatar CLI fallback to jpg frames: count=%s dir=%s", len(jpgs), frame_dir)
            for jpg in jpgs:
                container = av.open(str(jpg))
                try:
                    for frame in container.decode(video=0):
                        yield frame
                finally:
                    container.close()

    def _render_from_asset_video(
        self,
        pcm_s16le: bytes,
        sample_rate: int,
        avatar_asset: str,
    ) -> Iterator[VideoFrame]:
        duration_s = max(0.1, len(pcm_s16le) / float(max(1, sample_rate) * 2))
        target_frames = max(1, int(duration_s * self.fps))

        with tempfile.TemporaryDirectory(dir=self.temp_dir) as td:
            tmp_dir = Path(td)
            data_dir = Path(avatar_asset)
            if data_dir.suffix.lower() == ".zip":
                extracted_dir = tmp_dir / "avatar_data"
                extracted_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(data_dir, "r") as zf:
                    zf.extractall(extracted_dir)
                dirs = [p for p in extracted_dir.iterdir() if p.is_dir()]
                data_dir = dirs[0] if len(dirs) == 1 else extracted_dir

            candidates = [
                data_dir / "bg_video_palindrome.mp4",
                data_dir / "bg_video.mp4",
                data_dir / "bg_video_silence.mp4",
            ]
            video_path = next((p for p in candidates if p.exists()), None)
            if video_path is None:
                found = list(data_dir.rglob("*.mp4"))
                video_path = found[0] if found else None
            if video_path is None:
                raise RuntimeError("资产包内未找到可用背景视频(bg_video*.mp4)")

            decoded: list[VideoFrame] = []
            container = av.open(str(video_path))
            try:
                for frame in container.decode(video=0):
                    decoded.append(frame)
            finally:
                container.close()

            if not decoded:
                raise RuntimeError("背景视频可打开但无可解码帧")

            logger.info(
                "Avatar asset video fallback: video=%s decoded=%s target_frames=%s",
                video_path,
                len(decoded),
                target_frames,
            )

            count = len(decoded)
            for i in range(target_frames):
                yield decoded[self._pingpong_idx(i, count)]

    def _render_placeholder(self, pcm_s16le: bytes, sample_rate: int) -> Iterator[VideoFrame]:
        samples = np.frombuffer(pcm_s16le, dtype=np.int16)
        if samples.size == 0:
            samples = np.zeros(int(sample_rate * 0.5), dtype=np.int16)

        frame_count = max(1, int((samples.size / float(sample_rate)) * self.fps))
        samples_per_frame = max(1, int(sample_rate / self.fps))

        base = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        base[:, :, 0] = 18
        base[:, :, 1] = 18
        base[:, :, 2] = 24

        cx = self.width // 2
        cy = self.height // 2
        face_w = int(self.width * 0.24)
        face_h = int(self.height * 0.52)

        for idx in range(frame_count):
            left = idx * samples_per_frame
            right = min(samples.size, left + samples_per_frame)
            chunk = samples[left:right]
            amp = 0.0 if chunk.size == 0 else min(1.0, float(np.abs(chunk).mean() / 5000.0))

            canvas = base.copy()
            canvas[cy - face_h // 2 : cy + face_h // 2, cx - face_w // 2 : cx + face_w // 2, :] = [55, 60, 72]
            eye_y = cy - int(face_h * 0.18)
            eye_w = int(face_w * 0.12)
            eye_h = int(face_h * 0.04)
            canvas[eye_y : eye_y + eye_h, cx - int(face_w * 0.22) : cx - int(face_w * 0.22) + eye_w, :] = [220, 220, 230]
            canvas[eye_y : eye_y + eye_h, cx + int(face_w * 0.10) : cx + int(face_w * 0.10) + eye_w, :] = [220, 220, 230]

            mouth_base = int(face_h * 0.035)
            mouth_extra = int(face_h * 0.18 * amp)
            mouth_h = max(2, mouth_base + mouth_extra)
            mouth_w = int(face_w * 0.28)
            mouth_y = cy + int(face_h * 0.16)
            canvas[mouth_y : mouth_y + mouth_h, cx - mouth_w // 2 : cx + mouth_w // 2, :] = [245, 245, 245]

            frame = VideoFrame.from_ndarray(canvas, format="rgb24")
            yield frame

    def render(self, pcm_s16le: bytes, sample_rate: int, avatar_asset: str) -> Iterator[VideoFrame]:
        if self.lite_avatar_cli and not self._cli_disabled_reason:
            try:
                yield from self._render_with_lite_avatar_cli(
                    pcm_s16le=pcm_s16le,
                    sample_rate=sample_rate,
                    avatar_asset=avatar_asset,
                )
                return
            except Exception as exc:
                self._cli_disabled_reason = str(exc)
                logger.exception("Lite-Avatar render failed, fallback to asset video: %s", exc)
        elif self._cli_disabled_reason:
            logger.warning("Lite-Avatar CLI disabled for current process: %s", self._cli_disabled_reason[:300])

        try:
            yield from self._render_from_asset_video(
                pcm_s16le=pcm_s16le,
                sample_rate=sample_rate,
                avatar_asset=avatar_asset,
            )
            return
        except Exception as exc:
            logger.exception("Avatar asset video fallback failed, use placeholder: %s", exc)

        yield from self._render_placeholder(pcm_s16le=pcm_s16le, sample_rate=sample_rate)
