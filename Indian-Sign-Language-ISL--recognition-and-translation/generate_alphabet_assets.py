"""
Generate one stable reference image per A-Z ISL label from the bundled
training data (X_train.npy + y_train.npy). No external image folders needed.

For every label the script:
  1. Collects all training samples of that class.
  2. Averages them (per-pixel mean) - the training set is normalized
     [0, 1] grayscale, so the mean is a clean, stable "canonical" rendering.
  3. Rescales contrast to the full 0-255 range so the result is readable.
  4. Saves static/assets/isl_alphabet/<LETTER>.png (already 50x50; consumers
     resize to 100x100 when composing finger-spelling images).

Usage (from the repository root):
    python generate_alphabet_assets.py            # regenerate all 26 letters
    python generate_alphabet_assets.py --force    # same (kept for clarity)
    python generate_alphabet_assets.py --check    # exit 0 if all assets exist

The output folder is application-owned: regenerate any time the training data
changes. Images are deterministic given the same .npy files.
"""

import argparse
import sys

import cv2
import numpy as np

from isl_assets import ASSET_DIR, ISL_LABELS, assets_available


def generate_assets(output_dir=ASSET_DIR, verbose: bool = True) -> bool:
    """Generate one reference PNG per A-Z label. Returns True on success."""
    project_root = output_dir.parent.parent.parent  # static/assets/isl_alphabet -> repo root
    x_path = project_root / "X_train.npy"
    y_path = project_root / "y_train.npy"

    if not x_path.is_file() or not y_path.is_file():
        print(f"ERROR: training data not found at {x_path} / {y_path}", file=sys.stderr)
        return False

    if verbose:
        print(f"Loading {x_path.name} / {y_path.name} ...")
    X = np.load(x_path, mmap_mode="r")  # (N, 50, 50, 1) float32 in [0, 1]
    y = np.load(y_path, mmap_mode="r")  # (N, 26) one-hot

    if y.ndim == 2:  # one-hot -> class index
        y_idx = np.asarray(y).argmax(axis=1)
    else:
        y_idx = np.asarray(y).astype(int).ravel()

    if X.shape[0] != y_idx.shape[0]:
        print("ERROR: X and y sample counts differ", file=sys.stderr)
        return False

    if X.ndim == 4:
        images = X[:, :, :, 0]          # (N, 50, 50)
    elif X.ndim == 3:
        images = X
    else:
        print(f"ERROR: unexpected X shape {X.shape}", file=sys.stderr)
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    for label_index, letter in enumerate(ISL_LABELS):
        mask = y_idx == label_index
        if not mask.any():
            print(f"WARNING: no training samples for '{letter}' - skipped", file=sys.stderr)
            continue

        # Per-pixel mean of all samples for this class -> stable canonical image.
        mean_img = np.asarray(images[mask], dtype=np.float64).mean(axis=0)

        # Stretch contrast to full 0-255 for readability.
        lo, hi = float(mean_img.min()), float(mean_img.max())
        if hi > lo:
            mean_img = (mean_img - lo) / (hi - lo)
        uint8_img = (mean_img * 255.0).round().astype(np.uint8)

        out_path = output_dir / f"{letter}.png"
        cv2.imwrite(str(out_path), uint8_img)
        generated += 1
        if verbose:
            print(f"  {letter}: {int(mask.sum())} samples -> {out_path.relative_to(project_root)}")

    if verbose:
        print(f"Done: {generated}/{len(ISL_LABELS)} letters generated in {output_dir}")
    return generated == len(ISL_LABELS)


def main():
    parser = argparse.ArgumentParser(description="Generate A-Z ISL reference assets from training data")
    parser.add_argument("--check", action="store_true",
                        help="only verify that all 26 assets exist (exit 1 if not)")
    parser.add_argument("--force", action="store_true",
                        help="regenerate even if assets already exist (default behaviour)")
    args = parser.parse_args()

    if args.check:
        if assets_available():
            print("All 26 alphabet assets present.")
            sys.exit(0)
        missing = [l for l in ISL_LABELS
                   if not (ASSET_DIR / f"{l}.png").is_file()]
        print(f"Missing assets: {', '.join(missing)}", file=sys.stderr)
        print("Run: python generate_alphabet_assets.py", file=sys.stderr)
        sys.exit(1)

    ok = generate_assets()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
