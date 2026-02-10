# File: offline-avatar/apps/server/ws.py
from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from apps.server.session import Session
from modules.core import events
from modules.core.config import AppConfig

logger = logging.getLogger(__name__)


class WSApp:
    def __init__(self, config: AppConfig):
        self.config = config
        self.router = APIRouter()
        self.router.add_api_websocket_route("/ws", self.websocket_endpoint)

    async def websocket_endpoint(self, websocket: WebSocket) -> None:
        await websocket.accept()
        logger.info("WebSocket connected: client=%s", websocket.client)
        session = Session(config=self.config, send_json=websocket.send_json)

        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    message: dict[str, Any] = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON payload from client=%s", websocket.client)
                    continue

                msg_type = message.get("type")
                logger.info("WS message: type=%s client=%s", msg_type, websocket.client)

                try:
                    if msg_type == events.WS_TYPE_WEBRTC_OFFER:
                        answer = await session.handle_offer(
                            sdp=message.get("sdp", ""),
                            sdp_type=message.get("sdpType", "offer"),
                        )
                        await websocket.send_json(
                            {
                                "type": events.WS_TYPE_WEBRTC_ANSWER,
                                "sdp": answer["sdp"],
                                "sdpType": answer["sdpType"],
                            }
                        )
                        logger.info("WS answer sent: client=%s", websocket.client)
                    elif msg_type == events.WS_TYPE_WEBRTC_ICE:
                        await session.handle_ice(message.get("candidate"))
                    elif msg_type == events.WS_TYPE_INPUT_TEXT:
                        text = message.get("text", "")
                        logger.info("Text input length=%s", len(text or ""))
                        session.submit_text(text)
                    elif msg_type == events.WS_TYPE_INPUT_AUDIO:
                        data_base64 = message.get("data_base64", "")
                        logger.info(
                            "Audio input format=%s base64_len=%s",
                            message.get("format", "webm_opus"),
                            len(data_base64 or ""),
                        )
                        session.submit_audio(
                            fmt=message.get("format", "webm_opus"),
                            data_base64=data_base64,
                        )
                    elif msg_type == events.WS_TYPE_CHAT_CLEAR:
                        await session.clear_chat()
                except Exception:
                    logger.exception("WS message handling failed: type=%s", msg_type)
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected: client=%s", websocket.client)
        finally:
            await session.close()
            logger.info("Session closed: client=%s", websocket.client)
