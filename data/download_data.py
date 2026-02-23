#!/usr/bin/env python3
"""Telechargement du dataset NASA C-MAPSS (fourni - ne pas modifier)."""
import urllib.request
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw"

FILES = {
    "train_FD001.txt": "https://raw.githubusercontent.com/hankroark/Turbofan-Engine-Degradation/master/CMAPSSData/train_FD001.txt",
    "test_FD001.txt": "https://raw.githubusercontent.com/hankroark/Turbofan-Engine-Degradation/master/CMAPSSData/test_FD001.txt",
    "RUL_FD001.txt": "https://raw.githubusercontent.com/hankroark/Turbofan-Engine-Degradation/master/CMAPSSData/RUL_FD001.txt",
}

def download():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for fname, url in FILES.items():
        dest = RAW_DIR / fname
        if not dest.exists():
            print(f"  Telechargement {fname}...")
            urllib.request.urlretrieve(url, dest)
    print(f"Fichiers FD001 telecharges dans {RAW_DIR}")

if __name__ == "__main__":
    if (RAW_DIR / "train_FD001.txt").exists():
        print("Dataset deja present.")
    else:
        download()
