# File: offline-avatar/packages/core/events.py
WS_TYPE_WEBRTC_OFFER = "webrtc.offer"
WS_TYPE_WEBRTC_ANSWER = "webrtc.answer"
WS_TYPE_WEBRTC_ICE = "webrtc.ice"
WS_TYPE_INPUT_TEXT = "input.text"
WS_TYPE_INPUT_AUDIO = "input.audio"
WS_TYPE_CHAT_CLEAR = "chat.clear"

WS_TYPE_STATE = "state"
WS_TYPE_ASR_FINAL = "asr.final"
WS_TYPE_LLM_DELTA = "llm.delta"
WS_TYPE_LLM_FINAL = "llm.final"
WS_TYPE_CHAT_APPEND = "chat.append"
WS_TYPE_METRIC = "metric"

PHASE_IDLE = "idle"
PHASE_RECORDING = "recording"
PHASE_THINKING = "thinking"
PHASE_SPEAKING = "speaking"
