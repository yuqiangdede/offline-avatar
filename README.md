<!-- File: offline-avatar/README.md -->
# Offline Avatar (v0)

完全离线运行的数字人基础工程，包含：
- 前端 `React + Vite + TypeScript`
- 后端 `FastAPI + WebSocket + aiortc`
- Provider 抽象：`ASR / LLM / TTS / Avatar`
- 单一 WebSocket 复用：`WebRTC 信令 + 业务事件`

## 目录结构

```text
offline-avatar/
  apps/
    server/
    web/
  packages/
    core/
    providers/
  assets/
  configs/
  scripts/
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

## 后端启动（8000）

```bash
cd offline-avatar
git submodule update --init --recursive

# 创建虚拟环境（以下命令按系统二选一）
# Linux/macOS（使用 Python 3.10）
python3.10 -m venv .venv
source .venv/bin/activate

# Windows PowerShell（使用 Python 3.10，若无 py launcher 可用绝对路径）
# py -3.10 -m venv .venv
C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
# 可选：如果使用 CUDA 11.8（GPU）
pip install -r requirements.gpu.txt
python apps/server/main.py
```

如果需要 CUDA 11.8 版 GPU 依赖（PyTorch + ONNX Runtime），建议执行：

```bash
pip install -r requirements.gpu.txt
```

如果只用 CPU，请执行：

```bash
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1
```

健康检查：`http://localhost:8000/health`

说明：
- `requirements.txt` 已包含基础依赖与 Lite-Avatar 官方依赖（通过 `requirements.base.txt` + `models/lite-avatar/requirements.txt`）。
- `requirements.gpu.txt` 包含 CUDA 11.8 的 GPU 依赖（`torch/torchvision/torchaudio + onnxruntime-gpu`），请在 `requirements.txt` 之后安装。
- 若你只想先跑基础链路（不启用 Lite-Avatar），可执行：`pip install -r requirements.base.txt`

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

`configs/config.yaml` 支持切换 provider 与参数：
- `providers.asr`: `faster_whisper_small`
- `providers.llm`: `openai_compat_local`
- `providers.tts`: `pyttsx3`
- `providers.avatar`: `lite_avatar`
- `asr.device`: `cuda` / `cpu`
- `asr.device_index`: GPU 编号（多卡场景建议固定 `0`）
- `llm.endpoint`: 默认 `http://localhost:1234/api/v1/chat`
- `webrtc.video_codec`: 默认 `vp8`
- `webrtc.force_interface_ip`: 强制 WebRTC 使用指定本机网卡 IP（例如 `10.46.71.33`，会同时限制本地 ICE 采集与 SDP candidate）
- `webrtc.consent_interval_s` / `webrtc.consent_failures`: ICE consent 保活参数（长耗时渲染建议提高 `consent_failures`，如 `120`）
- `avatar.gpu_index`: Lite-Avatar 使用的 GPU 编号（默认 `0`）
- `avatar.ffmpeg_path`: ffmpeg 可执行文件路径（可指向项目内 `models/ffmpeg-*/bin/ffmpeg.exe`）
- `runtime.temp_dir`: 运行时临时目录（TTS wav、Lite-Avatar 中间文件等）
- `runtime.modelscope_cache_dir`: modelscope 缓存目录（避免写入用户主目录）

## Provider 标准协议（ASR / LLM / TTS）

完整协议已移至：`docs/provider-protocol.md`

## 说明与扩展

1. 当前 Avatar Provider 默认先使用占位渲染（口型条动画），可通过 `avatar.lite_avatar_cli` 配置 Lite-Avatar CLI 后启用真实渲染。
2. `assets/avatars/P1_4nURxeVKvzaVTealb-UJg.zip` 目前是占位文件，请替换为真实资源包。
3. ASR 使用 `faster-whisper`，首次加载模型请确保模型文件可在离线环境访问（建议预下载）。
4. 如需切换 TTS/ASR/LLM/Avatar，只需新增 provider 并在 `configs/config.yaml` 切换名称。
5. 使用 `models/lite-avatar` 子模块时，需下载权重（依赖已由根目录 `requirements.txt` 覆盖）：

```bash
# Windows
cd models/lite-avatar
download_model.bat
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
