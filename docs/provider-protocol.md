<!-- File: offline-avatar/docs/provider-protocol.md -->
# Provider 标准协议（ASR / LLM / TTS）

本文定义后端编排层与 Provider 的标准交互协议（v1）。后续新增实现请严格遵循该协议，避免改动 `apps/server/session.py` 主流程。

## 通用约束

- 语言码统一为 `zh` 或 `en`。
- Provider 发生错误时：抛出异常（`raise Exception(...)`），由编排层统一降级与告警。
- Provider 接口保持同步函数签名，由编排层通过 `asyncio.to_thread(...)` 异步调度。
- 所有返回值必须是可 JSON 序列化结构（除 `bytes` 二进制字段）。

## ASR 协议

接口定义：

```python
class ASRProvider:
    def transcribe(self, audio_bytes: bytes, sample_rate: int) -> dict
```

输入：

- `audio_bytes`: `PCM S16LE`、`mono`、小端字节流。
- `sample_rate`: 输入 PCM 采样率（当前主流程默认 16000）。

输出（必须字段）：

```json
{
  "text": "你好",
  "lang": "zh"
}
```

输出（可选字段）：

```json
{
  "segments": [
    { "start": 0.0, "end": 1.2, "text": "你好" }
  ]
}
```

约束：

- `text` 允许为空字符串（表示未识别到有效文本）。
- `lang` 必须归一化为 `zh|en`，不要返回 `zh-cn`、`en-us`。
- `segments` 若提供，`start/end` 使用秒（float）。

## LLM 协议

接口定义：

```python
class LLMProvider:
    def chat(self, messages: list, lang: str, stream: bool = False)
```

输入：

- `messages`: 对话上下文，OpenAI 风格结构：
  - `{"role":"user|assistant|system","content":"..."}`
- `lang`: 本轮目标回复语言（`zh|en`）。
- `stream`: 是否流式输出。

输出（非流式，`stream=False`）：

- 返回 `str`（完整回复文本）。

输出（流式，`stream=True`）：

- 返回 `iterator[str]`，每次 `yield` 一个增量文本片段（delta）。

约束：

- 即便上游接口是 OpenAI-compat，也要在 Provider 内完成字段适配（例如本地接口只接受 `model/system_prompt/input`）。
- 若响应结构复杂（如 `output` 为数组），Provider 必须在内部提取为最终 `str`。
- 返回空字符串表示“无可用回复”，不建议返回 `None`。

## TTS 协议

接口定义：

```python
class TTSProvider:
    def synthesize(self, text: str, lang: str) -> dict
```

输入：

- `text`: 要合成的单句或短文本。
- `lang`: 发音语言（`zh|en`）。

输出（必须字段）：

```json
{
  "pcm_s16le": "<bytes>",
  "sample_rate": 22050
}
```

约束：

- `pcm_s16le` 必须是 `PCM S16LE`、`mono`、小端。
- `sample_rate` 必须是正整数。
- 允许返回空音频（`pcm_s16le=b""`），编排层会跳过该句。
- 若输出采样率非 16k，编排层会重采样到 WebRTC 音轨采样率。

## 编排层调用顺序（规范）

语音输入：

1. `ASR.transcribe(...)`
2. `LLM.chat(..., stream=False|True)`
3. 按标点切句
4. 每句调用 `TTS.synthesize(...)`
5. 将每句音频送入 Avatar 渲染与 WebRTC 推流

文本输入：

1. 跳过 ASR
2. 直接进入 LLM -> 分句 -> TTS -> Avatar

## 新 Provider 接入验收清单

- ASR：返回 `text/lang` 且 `lang` 为 `zh|en`。
- LLM：非流式返回 `str`；流式返回 `iterator[str]`。
- TTS：返回 `pcm_s16le(bytes)` + `sample_rate(int)`。
- 异常路径：故障时抛异常，不吞错；日志包含错误原因。
- 实测：文本输入与语音输入各跑通 1 次，`chat.append` 与 `metric` 正常上报。
