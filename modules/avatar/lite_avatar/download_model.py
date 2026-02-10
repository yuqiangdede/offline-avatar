import json
import shutil
from pathlib import Path

import requests
import yaml
from modelscope.hub.api import HubApi
from modelscope.hub.file_download import get_file_download_url


LITE_AVATAR_MODEL_ID = "HumanAIGC-Engineering/LiteAvatarGallery"
PARAFORMER_MODEL_ID = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"

LITE_AVATAR_FILES = {
    "lite_avatar_weights/lm.pb": "weights/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/lm/lm.pb",
    "lite_avatar_weights/model_1.onnx": "weights/model_1.onnx",
    "lite_avatar_weights/model.pb": "weights/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/model.pb",
}

PARAFORMER_FILES = {
    "config.yaml": "weights/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/config.yaml",
    "am.mvn": "weights/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/am.mvn",
    "tokens.json": "weights/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/tokens.json",
    "seg_dict": "weights/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/seg_dict",
    "configuration.json": "weights/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/configuration.json",
}


def _build_info_map(api: HubApi, model_id: str, root: str | None = None) -> dict[str, dict]:
    files = api.get_model_files(model_id, root=root, recursive=True)
    return {item["Path"]: item for item in files if item.get("Type") == "blob"}


def _download_file(
    model_id: str,
    src_rel: str,
    dst_path: Path,
    info: dict,
    cache_dir: Path,
    session: requests.Session,
) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists() and info.get("Size") == dst_path.stat().st_size:
        print(f"Skip existing file: {dst_path}")
        return

    url = get_file_download_url(model_id, src_rel, info["Revision"])
    resp = session.get(url, timeout=600)
    resp.raise_for_status()

    cache_name = f"{Path(src_rel).name}.{info['Revision'][:8]}.bin"
    tmp_path = cache_dir / cache_name
    tmp_path.write_bytes(resp.content)
    shutil.copy2(tmp_path, dst_path)
    print(f"Downloaded {model_id}:{src_rel} -> {dst_path}")


def _patch_paraformer_config(config_path: Path, tokens_path: Path) -> None:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    name_map = {
        "model": {"Paraformer": "paraformer"},
        "encoder": {"SANMEncoder": "sanm"},
        "decoder": {"ParaformerSANMDecoder": "paraformer_decoder_sanm"},
        "predictor": {"CifPredictorV2": "cif_predictor_v2"},
        "frontend": {"WavFrontend": "wav_frontend"},
        "specaug": {"SpecAugLFR": "specaug_lfr"},
    }
    for field, field_map in name_map.items():
        val = cfg.get(field)
        if isinstance(val, str) and val in field_map:
            cfg[field] = field_map[val]

    cfg.setdefault("init", None)
    if "token_list" not in cfg:
        cfg["token_list"] = json.loads(tokens_path.read_text(encoding="utf-8"))

    config_path.write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"Patched paraformer config for local funasr compatibility: {config_path}")


def main() -> int:
    root = Path(__file__).resolve().parent
    cache_dir = root / ".modelscope_cache"
    cache_dir.mkdir(exist_ok=True)

    api = HubApi()
    session = requests.Session()
    session.trust_env = True

    lite_info = _build_info_map(api, LITE_AVATAR_MODEL_ID, root="lite_avatar_weights")
    para_info = _build_info_map(api, PARAFORMER_MODEL_ID)

    for src_rel, dst_rel in LITE_AVATAR_FILES.items():
        if src_rel not in lite_info:
            raise RuntimeError(f"Missing file in model repo {LITE_AVATAR_MODEL_ID}: {src_rel}")
        _download_file(
            LITE_AVATAR_MODEL_ID,
            src_rel,
            root / dst_rel,
            lite_info[src_rel],
            cache_dir,
            session,
        )

    for src_rel, dst_rel in PARAFORMER_FILES.items():
        if src_rel not in para_info:
            raise RuntimeError(f"Missing file in model repo {PARAFORMER_MODEL_ID}: {src_rel}")
        _download_file(
            PARAFORMER_MODEL_ID,
            src_rel,
            root / dst_rel,
            para_info[src_rel],
            cache_dir,
            session,
        )

    _patch_paraformer_config(
        root
        / "weights/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/config.yaml",
        root
        / "weights/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/tokens.json",
    )
    print("All model files downloaded successfully!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
