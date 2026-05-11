"""Download the FER2013 dataset.

The dataset is hosted in a few places. By default this tries the Hugging Face
mirror (no login required). If that fails it prints instructions for the
Kaggle alternative.

Usage:
    python scripts/download_fer2013.py [--out data]
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

HF_CSV_URL = "https://huggingface.co/datasets/Jeneral/fer-2013/resolve/main/fer2013.csv"
HF_CSV_FALLBACKS = [
    "https://huggingface.co/datasets/AKalo/fer-2013/resolve/main/fer2013.csv",
]


def download(url: str, dest: Path) -> bool:
    print(f"Downloading {url}\n  -> {dest}")
    try:
        with urllib.request.urlopen(url, timeout=60) as r, open(dest, "wb") as f:
            total = int(r.headers.get("Content-Length", 0))
            done = 0
            chunk = 1 << 15
            while True:
                buf = r.read(chunk)
                if not buf:
                    break
                f.write(buf); done += len(buf)
                if total:
                    sys.stdout.write(f"\r  {done/1e6:6.1f} / {total/1e6:6.1f} MB")
                    sys.stdout.flush()
            sys.stdout.write("\n")
        return True
    except Exception as e:
        print(f"  failed: {e}")
        if dest.exists() and dest.stat().st_size < 1024:
            dest.unlink()
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data", help="Output directory (default: data)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "fer2013.csv"
    if target.exists() and target.stat().st_size > 1_000_000:
        print(f"{target} already exists ({target.stat().st_size/1e6:.1f} MB). Nothing to do.")
        return

    for url in [HF_CSV_URL, *HF_CSV_FALLBACKS]:
        if download(url, target):
            print(f"\nDone. CSV saved to {target}.")
            return

    print(
        "\n"
        "Automatic download failed. Two manual options:\n"
        "  1) Kaggle:  https://www.kaggle.com/datasets/msambare/fer2013\n"
        "     Download, unzip into ./data/  (you'll get train/ and test/ folders).\n"
        "  2) Or place fer2013.csv directly in ./data/.\n"
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
