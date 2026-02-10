# ASR Model Download (faster-whisper-small)

This folder is a documentation anchor for ASR model setup.
The runtime model files should be placed at:

`models/faster-whisper-small`

Download command (run at project root):

```powershell
.\.venv\Scripts\Activate.ps1
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Systran/faster-whisper-small', local_dir='models/faster-whisper-small', local_dir_use_symlinks=False)"
```
