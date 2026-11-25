"""Copy Sawti's speech models to the owner's Hugging Face account.

Run by .github/workflows/mirror-models.yml. Needs HF_TOKEN (write access).
Every transcription model in the catalog (all languages) is copied at its
pinned revision into <HF_OWNER>/<repo-name> (public), keeping the README and
license files, the default quantization, Q4_K_M and every file under 500 MB.
`python mirror_to_hf.py --list` prints the model ids (used to build the
workflow matrix); ONLY=<id,...> limits a run to some models. The app verifies every file against the
catalog SHA-256, so the copies must be byte-identical (they are).
"""

import json
import os
import sys
from pathlib import Path


OWNER = os.environ.get("HF_OWNER", "GhaythMahadin")
CATALOG = Path(os.environ.get("CATALOG", "src-tauri/src/catalog/catalog.json"))

ROLES = ("asr", "langid")  # every transcription model + the language detector
SMALL_FILE = 500_000_000  # also copy every quant below 500 MB
TMP = Path("hf_tmp")


# Speaker-separation files the app tries from the owner's account first
# (src-tauri/src/model_choices.rs). Copied as job id "diarization".
EXTRAS = [
    ("csukuangfj/sherpa-onnx-pyannote-segmentation-3-0", "model.onnx"),
    ("csukuangfj/speaker-embedding-models", "3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx"),
    ("csukuangfj/speaker-embedding-models", "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"),
]


def model_ids(catalog: dict) -> list:
    return [i for i, m in catalog.items() if m["role"] in ROLES]


def mirror_extras(api, token) -> list:
    from huggingface_hub import hf_hub_download

    failures = []
    for src, f in EXTRAS:
        dst = f"{OWNER}/{src.split('/')[-1]}"
        print(f"\n=== diarization: {src}/{f} -> {dst}", flush=True)
        try:
            api.create_repo(dst, repo_type="model", private=False, exist_ok=True)
            if f in set(api.list_repo_files(dst)):
                print(f"  skip (already copied) {f}", flush=True)
                continue
            local = hf_hub_download(src, f, local_dir=TMP, token=token)
            api.upload_file(path_or_fileobj=local, path_in_repo=f, repo_id=dst,
                            commit_message=f"Mirror {f} from {src}")
            print(f"  copied {f}", flush=True)
        except Exception as e:
            print(f"  FAILED {f}: {e}", flush=True)
            failures.append(f"diarization/{f}")
    return failures


def main() -> int:
    catalog = {m["id"]: m for m in json.loads(CATALOG.read_text())["models"]}
    if "--list" in sys.argv:
        print(json.dumps(model_ids(catalog) + ["diarization"]))
        return 0
    from huggingface_hub import HfApi, hf_hub_download

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        print("HF_TOKEN is empty: add it under this repository's Settings > Secrets and variables > Actions.")
        return 1
    api = HfApi(token=token)
    only = [s for s in os.environ.get("ONLY", "").split(",") if s.strip()]
    failures = []
    if not only or "diarization" in only:
        failures += mirror_extras(api, token)
    for model_id in model_ids(catalog):
        if only and model_id not in only:
            continue
        m = catalog[model_id]
        src, rev = m["repository"], m["revision"]
        dst = f"{OWNER}/{src.split('/')[-1]}"
        print(f"\n=== {model_id}: {src}@{rev[:10]} -> {dst}", flush=True)
        api.create_repo(dst, repo_type="model", private=False, exist_ok=True)
        existing = set(api.list_repo_files(dst))
        sizes = {f["filename"]: f["size"] for f in m["files"].values()}
        wanted = []
        for f in api.list_repo_files(src, revision=rev):
            if f.endswith(".gguf"):
                size = sizes.get(f, 0)
                default = m["files"][m["quant"]]["filename"]
                if size and (size < SMALL_FILE or f == default or "Q4_K_M" in f):
                    wanted.append(f)
            elif not f.startswith(".") or f == ".gitattributes":
                wanted.append(f)
        for f in wanted:
            if f in existing and f.endswith(".gguf"):
                print(f"  skip (already copied) {f}", flush=True)
                continue
            try:
                local = hf_hub_download(src, f, revision=rev, local_dir=TMP, token=token)
                if f == "README.md":
                    text = Path(local).read_text(encoding="utf-8")
                    note = (
                        f"\n\n> Mirror of [{src}](https://huggingface.co/{src}) "
                        f"(revision `{rev}`) for the Sawti app. Original license applies.\n"
                    )
                    Path(local).write_text(text + note, encoding="utf-8")
                api.upload_file(
                    path_or_fileobj=local,
                    path_in_repo=f,
                    repo_id=dst,
                    commit_message=f"Mirror {f} from {src}@{rev[:10]}",
                )
                print(f"  copied {f}", flush=True)
            except Exception as e:  # keep going, report at the end
                print(f"  FAILED {f}: {e}", flush=True)
                failures.append(f"{model_id}/{f}")
            finally:
                for p in TMP.rglob("*"):
                    if p.is_file():
                        p.unlink()
    if failures:
        print("\nFailed files:\n  " + "\n  ".join(failures))
        return 1
    print("\nAll models mirrored.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
