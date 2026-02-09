# File: offline-avatar/apps/server/main.py
from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path

import aioice.ice as aioice_ice
import uvicorn
from av import logging as av_logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.server.ws import WSApp  # noqa: E402
from packages.core.config import ensure_project_dir, load_config  # noqa: E402

logger = logging.getLogger(__name__)
_AIOICE_ORIG_GET_HOST_ADDRESSES = aioice_ice.get_host_addresses
_AIOICE_PATCHED_FORCE_IP: str | None = None


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def setup_media_logging() -> None:
    # Suppress noisy FFmpeg swscale warnings (e.g. deprecated pixel format)
    # while keeping Python app logs at INFO level.
    av_logging.set_level(av_logging.ERROR)
    av_logging.set_libav_level(av_logging.ERROR)


def setup_runtime_env(config) -> None:
    temp_dir = ensure_project_dir(config, config.runtime.temp_dir)
    modelscope_cache = ensure_project_dir(config, config.runtime.modelscope_cache_dir)

    os.environ["TMP"] = temp_dir
    os.environ["TEMP"] = temp_dir
    os.environ["TMPDIR"] = temp_dir
    os.environ["MODELSCOPE_CACHE"] = modelscope_cache
    tempfile.tempdir = temp_dir

    logger.info(
        "Runtime dirs: temp_dir=%s, modelscope_cache=%s",
        temp_dir,
        modelscope_cache,
    )


def setup_webrtc_network_env(config) -> None:
    global _AIOICE_PATCHED_FORCE_IP
    force_ip = (config.webrtc.force_interface_ip or "").strip()
    consent_interval = max(1, int(config.webrtc.consent_interval_s))
    consent_failures = max(3, int(config.webrtc.consent_failures))

    aioice_ice.CONSENT_INTERVAL = consent_interval
    aioice_ice.CONSENT_FAILURES = consent_failures
    logger.info(
        "WebRTC consent config: interval_s=%s failures=%s (~%ss)",
        consent_interval,
        consent_failures,
        consent_interval * consent_failures,
    )

    if not force_ip:
        if _AIOICE_PATCHED_FORCE_IP is not None:
            aioice_ice.get_host_addresses = _AIOICE_ORIG_GET_HOST_ADDRESSES
            logger.info("WebRTC ICE host binding restored to default")
            _AIOICE_PATCHED_FORCE_IP = None
        return

    if _AIOICE_PATCHED_FORCE_IP == force_ip:
        return

    def _forced_get_host_addresses(use_ipv4: bool, use_ipv6: bool) -> list[str]:
        addresses = _AIOICE_ORIG_GET_HOST_ADDRESSES(use_ipv4=use_ipv4, use_ipv6=use_ipv6)
        if force_ip in addresses:
            return [force_ip]
        logger.warning(
            "WebRTC force_interface_ip=%s not found in local addresses=%s, fallback to default list",
            force_ip,
            addresses,
        )
        return addresses

    aioice_ice.get_host_addresses = _forced_get_host_addresses
    _AIOICE_PATCHED_FORCE_IP = force_ip
    logger.info("WebRTC ICE host binding forced to %s", force_ip)


def create_app() -> FastAPI:
    setup_logging()
    setup_media_logging()
    config = load_config(ROOT / "configs" / "config.yaml")
    setup_runtime_env(config)
    setup_webrtc_network_env(config)
    logger.info(
        "Server config loaded: providers(asr=%s,llm=%s,tts=%s,avatar=%s), llm_endpoint=%s, llm_model=%s, webrtc_force_ip=%s",
        config.providers.asr,
        config.providers.llm,
        config.providers.tts,
        config.providers.avatar,
        config.llm.endpoint,
        config.llm.model,
        config.webrtc.force_interface_ip or "-",
    )

    app = FastAPI(title="Offline Avatar", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    ws_app = WSApp(config=config)
    app.include_router(ws_app.router)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "apps.server.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
