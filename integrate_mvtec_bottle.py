#!/usr/bin/env python3
"""
Integrate MVTec AD bottle.zip into this project and retrain.

Usage:
  python integrate_mvtec_bottle.py /path/to/bottle.zip
  # or if already extracted:
  python integrate_mvtec_bottle.py /path/to/bottle_folder
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def extract_zip(zip_path: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest)
    # find bottle root (folder that contains train/)
    for p in dest.rglob("train"):
        if (p / "good").is_dir():
            return p.parent
    raise FileNotFoundError("Could not find train/good inside the zip")


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copytree(src, dst)


def setup_from_bottle_root(bottle: Path) -> None:
    train_good = bottle / "train" / "good"
    test_good = bottle / "test" / "good"
    if not train_good.is_dir():
        raise FileNotFoundError(f"Missing {train_good}")

    # clean data dirs
    for sub in ["train/good", "validation/good", "test/good", "test/defective"]:
        p = DATA / sub
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)

    # train = all good training images
    for f in sorted(train_good.glob("*")):
        if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
            shutil.copy2(f, DATA / "train" / "good" / f.name)

    # validation = half of test/good (if exists)
    goods = sorted(test_good.glob("*")) if test_good.is_dir() else []
    goods = [g for g in goods if g.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}]
    mid = max(1, len(goods) // 2)
    for f in goods[:mid]:
        shutil.copy2(f, DATA / "validation" / "good" / f.name)
    for f in goods[mid:]:
        shutil.copy2(f, DATA / "test" / "good" / f.name)

    # defective = all non-good test folders
    test_root = bottle / "test"
    if test_root.is_dir():
        for sub in test_root.iterdir():
            if sub.is_dir() and sub.name != "good":
                for f in sub.glob("*"):
                    if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                        shutil.copy2(f, DATA / "test" / "defective" / f"{sub.name}_{f.name}")

    n_train = len(list((DATA / "train/good").glob("*")))
    n_val = len(list((DATA / "validation/good").glob("*")))
    n_tg = len(list((DATA / "test/good").glob("*")))
    n_td = len(list((DATA / "test/defective").glob("*")))
    print(f"Dataset ready:")
    print(f"  train/good:        {n_train}")
    print(f"  validation/good:   {n_val}")
    print(f"  test/good:         {n_tg}")
    print(f"  test/defective:    {n_td}")
    print(f"  TOTAL:            {n_train + n_val + n_tg + n_td}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Path to bottle.zip OR extracted bottle/ folder")
    ap.add_argument("--train", action="store_true", help="Also train the model after setup")
    args = ap.parse_args()
    p = Path(args.path).expanduser().resolve()

    if p.suffix.lower() == ".zip":
        print(f"Extracting {p} ...")
        extract_dir = ROOT / "_mvtec_extract"
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        bottle = extract_zip(p, extract_dir)
        print(f"Found bottle root: {bottle}")
    else:
        bottle = p
        if not (bottle / "train" / "good").is_dir():
            # maybe path is parent
            cand = bottle / "bottle"
            if (cand / "train" / "good").is_dir():
                bottle = cand
            else:
                raise FileNotFoundError(f"No train/good under {p}")

    setup_from_bottle_root(bottle)

    if args.train:
        print("\nTraining model (good images only)...")
        from defect_hunter import train_default_model
        train_default_model(str(DATA), str(ROOT / "models" / "defect_hunter_model.pkl"))
        print("Model saved → models/defect_hunter_model.pkl")
    else:
        print("\nNext step: train the model with:")
        print("  python defect_hunter.py --data data --out models/defect_hunter_model.pkl")
        print("Then start the app:")
        print("  python app.py")


if __name__ == "__main__":
    main()
