<!-- File: offline-avatar/README.md -->
# Offline Avatar (v0)

完全离线运行的数字人基础工程，包含：
- 前端 `React + Vite + TypeScript`
- 后端 `FastAPI + WebSocket + aiortc`
- Provider 抽象：`ASR / LLM / TTS / Avatar`
- 单一 WebSocket 复用：`WebRTC 信令 + 业务事件`

## 目录结构

```text
digital-human/
  apps/
  modules/
    avatar/
      lite_avatar/
    asr/
      faster-whisper-small/
    tts/
      edge_tts/
    llm/
      openai_adapter/
  models/
    lite-avatar/
      P1_4nURxeVKvzaVTealb-UJg.zip
    faster-whisper-small/
  third_party/
    ffmpeg/
      bin/ffmpeg.exe
  runtime/
    tmp/
    cache/
    logs/
  configs/
    app.yaml
```

当前 Python 代码模块位于：

```text
modules/
    core/
    asr/faster_whisper/
    avatar/lite_avatar/
    llm/openai_adapter/
    tts/edge_tts/
```

## 功能说明

- 左侧视频区：播放 WebRTC 数字人画面，显示状态 `idle/recording/thinking/speaking`
- 右侧聊天面板：可折叠，显示用户/助手消息、语言、时间和延迟指标
- 底部输入栏：
  - 文本输入（Enter/按钮发送）
  - 录音输入（点击开始/停止）
- 后端流程：
  - `input.audio` -> 解码 -> ASR -> LLM -> 按句切分 -> TTS -> Avatar -> WebRTC A/V
  - `input.text` -> LLM -> TTS -> Avatar -> WebRTC A/V
- 语言策略：默认跟随输入语言（中/英），支持文本内指令覆盖（如“用英文回答”）

## 环境要求

- Python = 3.10
- CUDA = 11.8（仅在使用 GPU 时需要）
- Node.js 18+
- npm（可选 pnpm/yarn）
- 本地 ffmpeg（建议安装并加入 PATH）
- 本地 LLM HTTP 服务（OpenAI 兼容风格）

## 后端启动（GPU / 8000）

```bash
cd offline-avatar

# 创建虚拟环境（以下命令按系统二选一）
# Linux/macOS（使用 Python 3.10）
python3.10 -m venv .venv
source .venv/bin/activate

# Windows PowerShell（使用 Python 3.10，若无 py launcher 可用绝对路径）
# py -3.10 -m venv .venv
C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
pip install -r requirements.gpu.txt
```

### 模型准备（GPU 必做）

1. 下载 Lite-Avatar 权重（见 `modules/avatar/lite_avatar`，已包含 `speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch` 所需文件）：

```powershell
# Windows（在项目根目录 offline-avatar 下执行）
.\.venv\Scripts\Activate.ps1
cd .\modules\avatar\lite_avatar\
.\download_model.bat
cd ..\..\..\
```

2. 手动下载并解压 ffmpeg 到项目目录：
- 下载地址（Windows builds）：`https://www.gyan.dev/ffmpeg/builds/`
- 下载 `ffmpeg-n7.1-latest-win64-lgpl-7.1.zip`
- 解压到 `third_party/ffmpeg`
- 确认可执行文件存在：`third_party/ffmpeg/bin/ffmpeg.exe`

3. 下载 ASR 模型 `Systran/faster-whisper-small`（见 `modules/asr/faster-whisper-small/README.md`）：

```powershell
.\.venv\Scripts\Activate.ps1
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Systran/faster-whisper-small', local_dir='models/faster-whisper-small', local_dir_use_symlinks=False)"
```

4. 准备数字人资源包（模型目录）：

```powershell
# 将资源包放到 models/lite-avatar/P1_4nURxeVKvzaVTealb-UJg.zip
# 可选：提前解压到同级目录（便于检查文件）
Expand-Archive -Path .\models\lite-avatar\P1_4nURxeVKvzaVTealb-UJg.zip -DestinationPath .\models\lite-avatar\
```

### GitHub 提交边界（重要）

可上传到 GitHub（源码/配置）：
- `apps/`
- `modules/`（不含自动下载权重目录）
- `configs/app.yaml`
- `README.md`、`.gitignore`、`requirements*.txt`、`docs/`

不建议上传到 GitHub（本地下载产物）：
- `.venv/`、`runtime/`
- `third_party/ffmpeg/`
- `models/faster-whisper-small/`
- `models/lite-avatar/*.zip` 与解压目录（如 `models/lite-avatar/P1_*/`）
- `modules/avatar/lite_avatar/weights/` 与 `modules/avatar/lite_avatar/.modelscope_cache/`

本仓库已通过 `.gitignore` 排除上述下载产物；首次部署时请按下列步骤自行下载：
- Lite-Avatar 权重（含 `speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch`）：执行 `modules/avatar/lite_avatar/download_model.bat`
- ffmpeg：从 `https://www.gyan.dev/ffmpeg/builds/` 下载 `ffmpeg-n7.1-latest-win64-lgpl-7.1.zip`，解压到 `third_party/ffmpeg`
- ASR 模型：执行 `snapshot_download(repo_id='Systran/faster-whisper-small', local_dir='models/faster-whisper-small')`
- 数字人素材包：将私有资源包放到 `models/lite-avatar/P1_4nURxeVKvzaVTealb-UJg.zip`

5. 启动后端服务：

```powershell
.\.venv\Scripts\Activate.ps1
python apps/server/main.py
```

健康检查：`http://localhost:8000/health`

说明：
- `requirements.txt` 已包含基础依赖与 Lite-Avatar 官方依赖（通过 `requirements.base.txt` + `modules/avatar/lite_avatar/requirements.txt`）。
- `requirements.gpu.txt` 包含 CUDA 11.8 的 GPU 依赖（`torch/torchvision/torchaudio + onnxruntime-gpu`），请在 `requirements.txt` 之后安装。
- 本 README 当前仅覆盖 GPU 运行路径。

## 前端启动（5173）

```bash
cd apps/web
npm install
npm run dev
```

可选（如果已安装 pnpm）：

```bash
pnpm install
pnpm dev
```

浏览器访问：`http://localhost:5173`

## 配置文件

`configs/app.yaml` 支持切换 provider 与参数：
- `providers.asr`: `faster_whisper_small`
- `providers.llm`: `openai_compat_local`
- `providers.tts`: `pyttsx3`
- `providers.avatar`: `lite_avatar`
- `asr.device`: `cuda` / `cpu`
- `asr.model_size`: 默认 `models/faster-whisper-small`
- `asr.device_index`: GPU 编号（多卡场景建议固定 `0`）
- `llm.endpoint`: 默认 `http://localhost:1234/api/v1/chat`
- `webrtc.video_codec`: 默认 `vp8`
- `webrtc.force_interface_ip`: 强制 WebRTC 使用指定本机网卡 IP（例如 `10.46.71.33`，会同时限制本地 ICE 采集与 SDP candidate）
- `webrtc.consent_interval_s` / `webrtc.consent_failures`: ICE consent 保活参数（长耗时渲染建议提高 `consent_failures`，如 `120`）
- `avatar.gpu_index`: Lite-Avatar 使用的 GPU 编号（默认 `0`）
- `avatar.ffmpeg_path`: ffmpeg 可执行文件路径（默认 `third_party/ffmpeg/bin/ffmpeg.exe`）
- `avatar.delete_generated_mp4`: 是否删除本次生成 mp4（默认 `false` 不删除；保留到 `runtime/logs/generated-mp4/` 便于排查口型）
- `avatar.delete_generated_audio`: 是否删除本次生成音频（默认 `false` 不删除；保留到 `runtime/logs/generated-audio/` 便于核对口型）
- `avatar.short_chunk_merge_chars`: 分段后最短合并阈值（默认 `6`，过短分段自动与下一段合并）
- `avatar.lite_avatar_min_audio_ms`: Lite-Avatar 单段渲染最小音频时长（默认 `1800` ms，过短自动补尾部静音，仅用于渲染）
- `avatar.trim_tts_silence`: 是否启用 TTS 头尾静音裁剪（默认 `false`，避免误裁掉句首/句尾内容）
- `avatar.lite_avatar_render_sample_rate`: Lite-Avatar 渲染侧采样率（默认 `16000`，仅影响口型渲染输入，不影响前端播放采样率）
- `runtime.temp_dir`: 运行时临时目录（TTS wav、Lite-Avatar 中间文件等）
- `runtime.modelscope_cache_dir`: modelscope 缓存目录（避免写入用户主目录）

## Provider 标准协议（ASR / LLM / TTS）

完整协议已移至：`docs/provider-protocol.md`

## 说明与扩展

1. 当前 Avatar Provider 默认先使用占位渲染（口型条动画），可通过 `avatar.lite_avatar_cli` 配置 Lite-Avatar CLI 后启用真实渲染。
2. 头像资源默认读取 `models/lite-avatar/P1_4nURxeVKvzaVTealb-UJg.zip`。
3. ASR 使用 `faster-whisper`，请先将 `Systran/faster-whisper-small` 下载到 `models/faster-whisper-small`（见上方“模型准备”）。
4. 如需切换 TTS/ASR/LLM/Avatar，只需新增 provider 并在 `configs/app.yaml` 切换名称。
5. 使用 `modules/avatar/lite_avatar` 时，需执行 `download_model.bat` 下载权重；该步骤会同时准备 `speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch` 所需文件。

```powershell
# Windows（在项目根目录 offline-avatar 下执行）
.\.venv\Scripts\Activate.ps1
cd .\modules\avatar\lite_avatar\
.\download_model.bat
```

6. 若在 Windows 下 Lite-Avatar 未生成 mp4，本项目会自动回退读取 `tmp_frames/*.jpg` 生成 WebRTC 视频帧。
7. 若日志出现 `No module named 'xxx'`，请重新执行 `pip install -r requirements.txt` 后重启服务。

## WebSocket 协议（摘要）

客户端 -> 服务端：
- `webrtc.offer`
- `webrtc.ice`
- `input.text`
- `input.audio`
- `chat.clear`

服务端 -> 客户端：
- `webrtc.answer`
- `webrtc.ice`
- `state`
- `asr.final`
- `llm.final`
- `chat.append`
- `metric`

## 长句分段与排查参数

- 默认按标点切分 LLM 回复并逐段执行 `chat.append -> TTS -> Avatar`，前端会更早看到回复并开始播报，弱化长句场景下的等待和轻微音视频偏差体感。
- `avatar.single_pass_render`: 是否强制整段一次渲染（默认 `false`，即分段渲染）。
- `avatar.delete_generated_mp4`: 是否删除生成的 mp4（默认 `false`，保留到 `runtime/logs/generated-mp4/` 便于排查口型）。
- `avatar.delete_generated_audio`: 是否删除生成音频（默认 `false`，保留到 `runtime/logs/generated-audio/` 便于核对音视频）。
- `avatar.short_chunk_merge_chars`: 过短分段自动合并阈值（默认 `6`）。
- `avatar.lite_avatar_min_audio_ms`: Lite-Avatar 最小渲染音频时长（默认 `1800` ms）；短于阈值会自动补静音后再渲染，避免 `input_au ... Expected: 30` 这类短音频维度报错。
- `avatar.trim_tts_silence`: 默认关闭；若你确认 TTS 头尾空白较多且不会误伤语音内容，可再开启。
- `avatar.lite_avatar_render_sample_rate`: 建议保持 `16000`；参考 OpenAvatarChat 的“播放音频与算法音频分离”策略，渲染前统一重采样到 16k 可降低口型时序偏差。
