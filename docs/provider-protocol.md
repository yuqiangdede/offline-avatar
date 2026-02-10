<!-- File: offline-avatar/docs/provider-protocol.md -->
# Provider 标准协议（ASR / LLM / TTS）

本文定义 `apps/server/session.py` 与 Provider 的标准交互协议（v1.1）。  
目标是“新增/替换 Provider 不改主流程”，仅通过配置切换实现。

## 1. 适用范围

- 适用于 `modules/core/interfaces.py` 中的：
  - `ASRProvider`
  - `LLMProvider`
  - `TTSProvider`
- 不包含 `AvatarProvider`（其协议另行维护）。

## 2. 通用约束

- 语言码统一为 `zh` 或 `en`。
- 接口保持同步函数签名；由编排层通过 `asyncio.to_thread(...)` 调度。
- 出错必须抛异常（`raise`），不要吞错后返回伪成功结果。
- 文本字段必须是 `str`，不要返回 `None`。
- 除音频二进制字段（`bytes`）外，返回内容应为可序列化基础类型。
- Provider 内可以做上游协议适配，但对外必须满足本文返回约定。

## 3. ASR 协议

接口定义：

```python
class ASRProvider:
    def transcribe(self, audio_bytes: bytes, sample_rate: int) -> dict:
        ...
```

输入：

- `audio_bytes`: `PCM S16LE`、`mono`、little-endian 字节流。
- `sample_rate`: 输入采样率（当前主流程默认 `16000`）。

输出（必填）：

```json
{
  "text": "你好",
  "lang": "zh"
}
```

输出（可选）：

```json
{
  "segments": [
    { "start": 0.0, "end": 1.2, "text": "你好" }
  ]
}
```

约束：

- `text` 允许为空字符串（表示未识别到有效文本）。
- `lang` 必须归一化为 `zh|en`，不要返回 `zh-cn/en-us`。
- `segments` 若提供：
  - `start/end` 单位是秒（`float`）；
  - `text` 为 `str`；
  - 建议按时间递增。
- 建议空输入直接返回：`{"text": "", "lang": "zh", "segments": []}`。

## 4. LLM 协议

接口定义：

```python
class LLMProvider:
    def chat(self, messages: list, lang: str, stream: bool = False):
        ...
```

输入：

- `messages`: OpenAI 风格消息数组，元素结构：
  - `{"role": "user|assistant|system", "content": "..."}`
- `lang`: 目标回复语言（`zh|en`）。
- `stream`: 是否流式输出。

输出（`stream=False`）：

- 返回完整回复 `str`。

输出（`stream=True`）：

- 返回 `Iterator[str]`，每次 `yield` 增量文本（delta）。

约束：

- Provider 需要在内部适配不同上游返回结构（例如 `choices[]/output/text`）。
- 不返回 `None`；无可用内容时返回空字符串 `""`。
- 流式模式下，建议只 `yield` 非空增量字符串。

## 5. TTS 协议

接口定义：

```python
class TTSProvider:
    def synthesize(self, text: str, lang: str) -> dict:
        ...
```

输入：

- `text`: 句子或短文本。
- `lang`: 发音语言（`zh|en`）。

输出（必填）：

```json
{
  "pcm_s16le": "<bytes>",
  "sample_rate": 22050
}
```

约束：

- `pcm_s16le` 必须为 `PCM S16LE`、`mono`、little-endian。
- `sample_rate` 必须是正整数。
- 空文本可返回空音频：`{"pcm_s16le": b"", "sample_rate": 16000}`。
- 编排层会处理重采样与降级，不要求 TTS 端固定 16k。

## 6. 与当前编排层实现的对齐点

以 `apps/server/session.py` 当前逻辑为准：

- ASR：
  - 主流程只强依赖 `text/lang`；
  - `segments` 为可选扩展字段。
- LLM：
  - 当前主流程调用 `chat(..., stream=False)`；
  - `llm.delta` 事件预留，后续可接流式。
- TTS：
  - 主流程只依赖 `pcm_s16le/sample_rate`；
  - 空音频会触发降级策略（备用语音/提示音）。

## 7. Provider 标识与配置对齐

`configs/app.yaml` 中已对齐如下标识：

- `providers.asr: faster_whisper_small`
- `providers.llm: openai_compat_local`
- `providers.tts: pyttsx3`

新增 Provider 时，需同时完成：

1. 新实现类（`modules/.../provider.py`）
2. 构建函数分支（`apps/server/session.py` 中 `_build_*_provider`）
3. `configs/app.yaml` 的 provider 名称
4. README 使用说明

## 8. 接入验收清单

- ASR：返回 `text/lang`，且 `lang in {"zh","en"}`。
- LLM：非流式返回 `str`；流式返回 `Iterator[str]`。
- TTS：返回 `pcm_s16le(bytes)` 与 `sample_rate(int)`。
- 异常路径：故障时抛异常，日志可定位原因。
- 联调通过：
  - 文本输入链路（`input.text`）跑通；
  - 语音输入链路（`input.audio`）跑通；
  - 前端可收到 `chat.append` / `metric` / `llm.final`（以及 ASR 场景的 `asr.final`）。
