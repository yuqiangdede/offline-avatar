import shutil
from pathlib import Path

from modelscope.hub.api import HubApi, ModelScopeConfig
from modelscope.hub.file_download import get_file_download_url, http_get_file
from modelscope.hub.utils.utils import file_integrity_validation


MODEL_ID = "HumanAIGC-Engineering/LiteAvatarGallery"
FILES = {
    "lite_avatar_weights/lm.pb": "weights/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/lm/lm.pb",
    "lite_avatar_weights/model_1.onnx": "weights/model_1.onnx",
    "lite_avatar_weights/model.pb": "weights/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/model.pb",
}


def main() -> int:
    root = Path(__file__).resolve().parent
    cache_dir = root / ".modelscope_cache"
    cache_dir.mkdir(exist_ok=True)

    api = HubApi()
    file_infos = api.get_model_files(
        MODEL_ID, root="lite_avatar_weights", recursive=True
    )
    info_map = {info["Path"]: info for info in file_infos if info["Type"] == "blob"}
    cookies = ModelScopeConfig.get_cookies()

    for src_rel, dst_rel in FILES.items():
        if src_rel not in info_map:
            raise RuntimeError(f"Missing file in model repo: {src_rel}")
        info = info_map[src_rel]
        dst_path = root / dst_rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if dst_path.exists() and info.get("Size") == dst_path.stat().st_size:
            print(f"Skip existing file: {dst_path}")
            continue

        url = get_file_download_url(MODEL_ID, src_rel, info["Revision"])
        tmp_name = f"{Path(info['Name']).stem}.{info['Revision'][:8]}.bin"
        http_get_file(url, str(cache_dir), tmp_name, cookies)
        tmp_path = cache_dir / tmp_name
        if info.get("Sha256"):
            file_integrity_validation(str(tmp_path), info["Sha256"])

        shutil.copy2(tmp_path, dst_path)
        print(f"Copied {tmp_path} -> {dst_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
