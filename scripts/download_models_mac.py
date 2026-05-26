#!/usr/bin/env python3
"""
One-time download of Mobile-O weights into the repo (for faster local loading).

Downloads:
  1. Amshaker/Mobile-O-0.5B  -> models/Mobile-O-0.5B/   (~4.8 GB)
  2. SANA DiT+VAE (used at first inference) -> HF cache only unless --local-sana

After this, use:
  --model-path models/Mobile-O-0.5B

Run once:
  python scripts/download_models_mac.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MOBILEO_DIR = REPO_ROOT / "models" / "Mobile-O-0.5B"
SANA_REPO = "Efficient-Large-Model/Sana_600M_512px_diffusers"


def download_mobileo(local_dir: Path) -> Path:
    from huggingface_hub import snapshot_download

    print(f"Downloading {local_dir.name} (~4.8 GB) …")
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id="Amshaker/Mobile-O-0.5B",
        repo_type="model",
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    weights = local_dir / "model.safetensors"
    if weights.exists():
        size_gb = weights.stat().st_size / (1024**3)
        print(f"  OK: {weights} ({size_gb:.2f} GB)")
        if size_gb < 1.0:
            print("  ERROR: file too small — likely an LFS pointer. Re-run or use huggingface-cli.", file=sys.stderr)
            sys.exit(1)
    else:
        # Merged subfolder layout from some download scripts
        merged = list(local_dir.glob("**/model.safetensors"))
        if not merged:
            print("  WARNING: model.safetensors not found at repo root; check layout.", file=sys.stderr)
    return Path(path)


def download_sana_to_cache() -> None:
    from huggingface_hub import snapshot_download

    print(f"Pre-caching {SANA_REPO} (DiT + VAE + scheduler) …")
    snapshot_download(repo_id=SANA_REPO, repo_type="model")
    print("  OK: SANA cached under ~/.cache/huggingface/hub/")


def main() -> None:
    p = argparse.ArgumentParser(description="Download Mobile-O models for local Mac use")
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_MOBILEO_DIR),
        help=f"Local folder for Mobile-O-0.5B (default: {DEFAULT_MOBILEO_DIR})",
    )
    p.add_argument(
        "--skip-sana",
        action="store_true",
        help="Do not pre-download SANA (will download on first inference instead)",
    )
    args = p.parse_args()

    out = Path(args.output_dir).resolve()
    print(f"Repo root: {REPO_ROOT}\n")

    download_mobileo(out)

    if not args.skip_sana:
        download_sana_to_cache()

    print("\nDone. Use this path for inference and benchmark:")
    print(f"  --model-path {out.relative_to(REPO_ROOT)}")
    print("\nExample:")
    print(f"  python scripts/run_mobileo_benchmark_mac.py --model-path {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
