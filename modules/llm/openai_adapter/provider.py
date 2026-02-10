# File: offline-avatar/modules/llm/openai_adapter/provider.py
from __future__ import annotations

import json
import logging
from typing import Generator

import requests

from modules.core.interfaces import LLMProvider

logger = logging.getLogger(__name__)


class OpenAICompatLocalProvider(LLMProvider):
    def __init__(
        self,
        endpoint: str,
        model: str,
        timeout_s: int = 60,
        system_prompt_zh: str = "你是一个离线数字人助手。请简洁、准确地回答。",
        system_prompt_en: str = "You are an offline digital human assistant. Reply concisely and accurately.",
    ):
        self.endpoint = endpoint
        self.model = model
        self.timeout_s = timeout_s
        self.system_prompt_zh = system_prompt_zh
        self.system_prompt_en = system_prompt_en

    def _build_payload(self, messages: list, lang: str, stream: bool) -> dict:
        system_prompt = self.system_prompt_zh if lang == "zh" else self.system_prompt_en
        user_input = ""
        for item in reversed(messages):
            if item.get("role") == "user":
                user_input = item.get("content", "")
                break

        payload = {
            "model": self.model,
            "system_prompt": system_prompt,
            "input": user_input,
        }
        # Some OpenAI-compatible local endpoints (e.g. LM Studio /api/v1/chat)
        # reject unknown keys. Only send "stream" when explicitly enabled.
        if stream:
            payload["stream"] = True
        return payload

    @staticmethod
    def _extract_text(data: dict) -> str:
        if isinstance(data.get("output"), str):
            return data["output"]
        if isinstance(data.get("output"), list):
            chunks: list[str] = []
            for item in data["output"]:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if isinstance(content, str):
                    chunks.append(content)
            if chunks:
                return "\n".join(chunks)
        if isinstance(data.get("text"), str):
            return data["text"]

        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0] or {}
            message = first.get("message") or {}
            if isinstance(message.get("content"), str):
                return message["content"]
            if isinstance(first.get("text"), str):
                return first["text"]

        result = data.get("result")
        if isinstance(result, str):
            return result

        return ""

    def _stream_chat(self, payload: dict) -> Generator[str, None, None]:
        logger.info(
            "LLM(stream) request: endpoint=%s model=%s lang=%s input=%s",
            self.endpoint,
            self.model,
            payload.get("lang"),
            (payload.get("input") or "")[:80],
        )
        with requests.post(
            self.endpoint,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout_s,
            stream=True,
        ) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = raw_line.strip()
                if line.startswith("data:"):
                    line = line[5:].strip()
                if line == "[DONE]":
                    break
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                delta = ""
                if isinstance(obj.get("delta"), str):
                    delta = obj["delta"]
                elif isinstance(obj.get("text"), str):
                    delta = obj["text"]
                else:
                    choices = obj.get("choices")
                    if isinstance(choices, list) and choices:
                        choice = choices[0] or {}
                        c_delta = choice.get("delta") or {}
                        if isinstance(c_delta.get("content"), str):
                            delta = c_delta["content"]

                if delta:
                    yield delta

    def chat(self, messages: list, lang: str, stream: bool = False):
        payload = self._build_payload(messages=messages, lang=lang, stream=stream)

        if stream:
            return self._stream_chat(payload)

        logger.info(
            "LLM request: endpoint=%s model=%s lang=%s messages=%s input=%s",
            self.endpoint,
            self.model,
            lang,
            len(messages),
            (payload.get("input") or "")[:120],
        )

        try:
            response = requests.post(
                self.endpoint,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout_s,
            )
        except requests.RequestException:
            logger.exception("LLM HTTP request failed: endpoint=%s", self.endpoint)
            raise

        if not response.ok:
            logger.error(
                "LLM HTTP status error: status=%s body=%s",
                response.status_code,
                (response.text or "")[:500],
            )
            response.raise_for_status()

        try:
            data = response.json()
        except ValueError:
            logger.exception("LLM JSON decode failed: body=%s", (response.text or "")[:500])
            raise

        text = self._extract_text(data).strip()
        logger.info("LLM response parsed: text_len=%s", len(text))
        if not text:
            logger.warning("LLM response has no usable text: keys=%s", list(data.keys()))
        return text
