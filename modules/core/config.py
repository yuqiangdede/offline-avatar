# File: offline-avatar/modules/core/config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ProvidersConfig:
    asr: str = "faster_whisper_small"
    llm: str = "openai_compat_local"
    tts: str = "pyttsx3"
    avatar: str = "lite_avatar"


@dataclass
class AudioConfig:
    sample_rate: int = 16000


@dataclass
class WebRTCConfig:
    video_codec: str = "vp8"
    fps: int = 20
    width: int = 1280
    height: int = 720
    force_interface_ip: str = ""
    consent_interval_s: int = 5
    consent_failures: int = 120


@dataclass
class LLMConfig:
    endpoint: str = "http://localhost:1234/api/v1/chat"
    model: str = "qwen/qwen3-4b-2507"
    timeout_s: int = 60
    system_prompt_zh: str = "你是一个离线数字人助手。请简洁、准确地回答。尽量不要使用表情符号。"
    system_prompt_en: str = "You are an offline digital human assistant. Reply concisely and accurately."


@dataclass
class ASRConfig:
    model_size: str = "models/faster-whisper-small"
    device: str = "cpu"
    compute_type: str = "int8"
    device_index: int = 0


@dataclass
class TTSConfig:
    rate: int = 185
    volume: float = 1.0


@dataclass
class AvatarConfig:
    default_asset: str = "models/lite-avatar/P1_4nURxeVKvzaVTealb-UJg.zip"
    fallback_resolution: str = "720p"
    lite_avatar_cli: str = ""
    gpu_index: int = 0
    ffmpeg_path: str = ""
    delete_generated_mp4: bool = False
    delete_generated_audio: bool = False
    single_pass_render: bool = False
    short_chunk_merge_chars: int = 6
    lite_avatar_min_audio_ms: int = 1800
    trim_tts_silence: bool = False
    lite_avatar_render_sample_rate: int = 16000


@dataclass
class RuntimeConfig:
    temp_dir: str = "runtime/tmp"
    modelscope_cache_dir: str = "runtime/cache/modelscope"


@dataclass
class AppConfig:
    providers: ProvidersConfig
    audio: AudioConfig
    webrtc: WebRTCConfig
    llm: LLMConfig
    asr: ASRConfig
    tts: TTSConfig
    avatar: AvatarConfig
    runtime: RuntimeConfig
    project_root: Path


_DEFAULTS: dict[str, Any] = {
    "providers": {
        "asr": "faster_whisper_small",
        "llm": "openai_compat_local",
        "tts": "pyttsx3",
        "avatar": "lite_avatar",
    },
    "audio": {"sample_rate": 16000},
    "webrtc": {
        "video_codec": "vp8",
        "fps": 20,
        "width": 1280,
        "height": 720,
        "force_interface_ip": "",
        "consent_interval_s": 5,
        "consent_failures": 120,
    },
    "llm": {
        "endpoint": "http://localhost:1234/api/v1/chat",
        "model": "qwen/qwen3-4b-2507",
        "timeout_s": 60,
    },
    "asr": {"model_size": "models/faster-whisper-small", "device": "cpu", "compute_type": "int8", "device_index": 0},
    "tts": {"rate": 185, "volume": 1.0},
    "avatar": {
        "default_asset": "models/lite-avatar/P1_4nURxeVKvzaVTealb-UJg.zip",
        "fallback_resolution": "720p",
        "lite_avatar_cli": "",
        "gpu_index": 0,
        "ffmpeg_path": "",
        "delete_generated_mp4": False,
        "delete_generated_audio": False,
        "single_pass_render": False,
        "short_chunk_merge_chars": 6,
        "lite_avatar_min_audio_ms": 1800,
        "trim_tts_silence": False,
        "lite_avatar_render_sample_rate": 16000,
    },
    "runtime": {
        "temp_dir": "runtime/tmp",
        "modelscope_cache_dir": "runtime/cache/modelscope",
    },
}


def _merge_dict(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path = "configs/app.yaml") -> AppConfig:
    path_obj = Path(path).resolve()
    project_root = path_obj.parent.parent

    loaded: dict[str, Any] = {}
    if path_obj.exists():
        loaded = yaml.safe_load(path_obj.read_text(encoding="utf-8")) or {}

    merged = _merge_dict(_DEFAULTS, loaded)

    providers = ProvidersConfig(**merged["providers"])
    audio = AudioConfig(**merged["audio"])
    webrtc = WebRTCConfig(**merged["webrtc"])
    llm = LLMConfig(**merged["llm"])
    asr = ASRConfig(**merged["asr"])
    tts = TTSConfig(**merged["tts"])
    avatar = AvatarConfig(**merged["avatar"])
    runtime = RuntimeConfig(**merged["runtime"])

    return AppConfig(
        providers=providers,
        audio=audio,
        webrtc=webrtc,
        llm=llm,
        asr=asr,
        tts=tts,
        avatar=avatar,
        runtime=runtime,
        project_root=project_root,
    )


def resolve_project_path(config: AppConfig, maybe_relative_path: str) -> str:
    path = Path(maybe_relative_path)
    if path.is_absolute():
        return str(path)
    return str((config.project_root / path).resolve())


def ensure_project_dir(config: AppConfig, maybe_relative_dir: str) -> str:
    path = Path(resolve_project_path(config, maybe_relative_dir))
    path.mkdir(parents=True, exist_ok=True)
    return str(path)
