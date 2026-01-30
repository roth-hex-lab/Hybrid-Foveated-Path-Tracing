#!/usr/bin/env python3
"""
Merge COLMAP datasets with flexible modes and robust duplicate checks.

Supports two scenarios:
1) TRAIN + GROUND-TRUTH (GT):
   - Copies GT images into the train dataset, appends entries to train's images.txt.
   - Writes/extends/replaces test.txt with the (possibly renamed) GT image names.

2) TRAIN + TRAIN:
   - Merges additional train datasets into the first train dataset.
   - No test.txt is written/modified.

Duplicate detection is based on camera pose (qvec + tvec) from images.txt, NOT filenames.
File name collisions are resolved by auto-renaming and updating images.txt/test.txt accordingly.

Notes / Assumptions:
- This script only touches `images/` and `sparse/0/images.txt` (and `test.txt`). It does NOT
  merge/modify cameras.txt or points3D.txt. If the merged datasets define cameras inconsistently,
  downstream tools may need you to reconcile cameras separately.
"""

from __future__ import annotations
import os
import shutil
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Iterable, Dict, Optional, Set

# ----------------------------
# Data structures & I/O utils
# ----------------------------

@dataclass
class ColmapImage:
    """Represents a single two-line entry in COLMAP images.txt.

    Attributes
    ----------
    id : int
        Image ID in images.txt.
    qvec : List[float]
        Quaternion (qw, qx, qy, qz) as in COLMAP.
    tvec : List[float]
        Translation vector (tx, ty, tz) as in COLMAP.
    camera_id : int
        Camera ID reference from images.txt.
    name : str
        Filename of the image.
    points_line : str
        The second line of the entry (2D keypoints + point3D associations) kept as-is.
    is_comment : bool
        True if this object is a comment line, not an image entry.
    content : str
        Raw comment line content (for comments only).
    """
    id: int = -1
    qvec: List[float] = None
    tvec: List[float] = None
    camera_id: int = -1
    name: str = ""
    points_line: str = ""
    is_comment: bool = False
    content: str = ""

    def to_string(self) -> str:
        if self.is_comment:
            # Preserve comments as single-line entries ending with a newline
            return self.content.rstrip("\n") + "\n"
        qvec_str = " ".join(map(str, self.qvec))
        tvec_str = " ".join(map(str, self.tvec))
        line1 = f"{self.id} {qvec_str} {tvec_str} {self.camera_id} {self.name}"
        return f"{line1}\n{self.points_line.strip()}\n"

    def pose_tuple(self) -> Tuple[float, float, float, float, float, float, float]:
        return tuple(self.qvec + self.tvec)


def parse_images_txt(path: str) -> Tuple[List[ColmapImage], int]:
    """Parse a COLMAP images.txt file.

    Returns
    -------
    images : list of ColmapImage
        Includes both comments and actual image entries (comments first, typically).
    max_id : int
        Maximum image id found among entries (0 if none).
    """
    images: List[ColmapImage] = []
    max_id = 0

    try:
        with open(path, "r") as fid:
            while True:
                line = fid.readline()
                if not line:
                    break
                line_stripped = line.strip()
                if not line_stripped:
                    # blank lines: skip writing them back to avoid format surprises
                    continue
                if line_stripped.startswith("#"):
                    images.append(ColmapImage(is_comment=True, content=line))
                    continue

                # image line
                elems = line_stripped.split()
                if len(elems) < 10:
                    raise IOError(
                        f"Malformed images.txt line at {path}: '{line_stripped}'"
                    )
                image_id = int(elems[0])
                qvec = list(map(float, elems[1:5]))
                tvec = list(map(float, elems[5:8]))
                camera_id = int(elems[8])
                image_name = elems[9]

                max_id = max(max_id, image_id)

                # next line is points
                points_line = fid.readline()
                if not points_line:
                    raise IOError(
                        f"File {path} ended unexpectedly. Missing points for image {image_id}."
                    )

                images.append(
                    ColmapImage(
                        id=image_id,
                        qvec=qvec,
                        tvec=tvec,
                        camera_id=camera_id,
                        name=image_name,
                        points_line=points_line,
                    )
                )
    except Exception as e:
        raise RuntimeError(f"Error reading images.txt at '{path}': {e}")

    return images, max_id


def write_images_txt(path: str, images: List[ColmapImage]) -> None:
    try:
        with open(path, "w") as fid:
            for img in images:
                fid.write(img.to_string())
    except Exception as e:
        raise RuntimeError(f"Error writing images.txt at '{path}': {e}")


# ----------------------------
# Helpers
# ----------------------------

def ensure_exists(path: str, kind: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required {kind} not found: {path}")


def build_pose_key(vals: Iterable[float], tol: float) -> Tuple[int, ...]:
    """Quantize floats by tolerance and return an int tuple for stable hashing."""
    return tuple(int(round(v / tol)) for v in vals)


def pose_keys_from_images(images: Iterable[ColmapImage], tol: float) -> Dict[Tuple[int, ...], ColmapImage]:
    keys: Dict[Tuple[int, ...], ColmapImage] = {}
    for im in images:
        if im.is_comment:
            continue
        keys[build_pose_key(im.pose_tuple(), tol)] = im
    return keys


def next_unique_name(dst_dir: str, filename: str) -> str:
    """Return a filename that does not exist in dst_dir by appending _N before extension."""
    base = os.path.basename(filename)
    name, ext = os.path.splitext(base)
    candidate = base
    idx = 2
    while os.path.exists(os.path.join(dst_dir, candidate)):
        candidate = f"{name}_{idx}{ext}"
        idx += 1
    return candidate


def copy_images_with_renaming(src_dir: str, dst_dir: str, names: Iterable[str]) -> Dict[str, str]:
    """Copy images from src_dir to dst_dir; rename on collision.

    Returns a mapping {original_name_in_images_txt -> final_copied_name}.
    If a name already exists in dst_dir, we create a new unique name.
    """
    mapping: Dict[str, str] = {}
    os.makedirs(dst_dir, exist_ok=True)

    for name in names:
        src = os.path.join(src_dir, name)
        if not os.path.isfile(src):
            # Warn but proceed (images.txt may reference missing files)
            print(f"[WARN] Source image missing: {src}")
            # still record mapping to same name for consistency
            mapping[name] = name
            continue
        # Resolve destination filename (rename if exists)
        final_name = name
        dst = os.path.join(dst_dir, final_name)
        if os.path.exists(dst):
            final_name = next_unique_name(dst_dir, name)
            dst = os.path.join(dst_dir, final_name)
        shutil.copy2(src, dst)
        mapping[name] = final_name
    return mapping


def apply_name_mapping(images: Iterable[ColmapImage], mapping: Dict[str, str]) -> None:
    for im in images:
        if im.is_comment:
            continue
        if im.name in mapping:
            im.name = mapping[im.name]


def read_test_txt(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]


def write_test_txt(path: str, names: Iterable[str]) -> None:
    with open(path, "w") as f:
        for n in names:
            f.write(n + "\n")


# ---------------------------------
# Core merge operations
# ---------------------------------

def merge_into_train(
    train_dir: str,
    other_dir: str,
    *,
    is_gt: bool,
    gt_mode: str,
    pose_tol: float,
    duplicate_resolution: str,
) -> None:
    """Merge `other_dir` dataset into `train_dir`.

    Parameters
    ----------
    train_dir : str
        Base dataset to extend.
    other_dir : str
        Dataset to merge into base.
    is_gt : bool
        If True, treat as ground-truth merge and manage test.txt per gt_mode.
    gt_mode : {"error", "extend", "replace"}
        Behavior when test.txt already exists (GT already added before).
    pose_tol : float
        Tolerance used for pose-based duplicate detection.
    """
    train_images_dir = os.path.join(train_dir, "images")
    train_sparse_dir = os.path.join(train_dir, "sparse", "0")
    train_images_txt = os.path.join(train_sparse_dir, "images.txt")
    test_txt = os.path.join(train_sparse_dir, "test.txt")

    other_images_dir = os.path.join(other_dir, "images")
    other_images_txt = os.path.join(other_dir, "sparse", "0", "images.txt")

    # Validate existence
    ensure_exists(train_images_dir, "train images dir")
    ensure_exists(train_images_txt, "train images.txt")
    ensure_exists(other_images_dir, "other images dir")
    ensure_exists(other_images_txt, "other images.txt")

    # Read both images.txt
    train_images_all, train_max_id = parse_images_txt(train_images_txt)
    other_images_all, _ = parse_images_txt(other_images_txt)

    train_entries = [im for im in train_images_all if not im.is_comment]
    other_entries = [im for im in other_images_all if not im.is_comment]

    print(f"Merging into: {train_dir}")
    print(f" - Existing train images: {len(train_entries)} (max id {train_max_id})")
    print(f" - Incoming images from {other_dir}: {len(other_entries)})")

    # If GT merge, handle test.txt existence according to mode
    if is_gt and os.path.exists(test_txt):
        if gt_mode == "error":
            print(
                "[ERROR] test.txt already exists; this dataset already has GT assigned. "
                "Use --gt_mode extend or replace."
            )
            raise SystemExit(1)
        elif gt_mode not in ("extend", "replace"):
            print(f"[ERROR] Invalid gt_mode: {gt_mode}")
            raise SystemExit(2)
    
    # Pose-based duplicate detection
    train_pose_keys = pose_keys_from_images(train_entries, pose_tol)
    other_pose_keys = pose_keys_from_images(other_entries, pose_tol)
    
    duplicate_keys = set(train_pose_keys.keys()).intersection(set(other_pose_keys.keys()))
    if duplicate_keys:
        if duplicate_resolution == "error":
            print(f"[ERROR] Attempting to add {len(duplicate_keys)} image(s) with identical pose (qvec+tvec) as existing entries.")
            raise SystemExit(1)
        if duplicate_resolution == "remaining":
            other_entries = [im[1] for im in other_pose_keys.items() if im[0] not in duplicate_keys]
            print(f"Reduced new entries from {len(other_pose_keys)} to {len(other_entries)} due to duplicates")


    # Copy images with rename collision handling
    incoming_names = [im.name for im in other_entries]
    name_map = copy_images_with_renaming(other_images_dir, train_images_dir, incoming_names)

    # Update image names post-rename
    apply_name_mapping(other_entries, name_map)

    # test.txt management for GT merge
    if is_gt:
        # Names used for test list are the final (possibly renamed) filenames
        new_test_names = [im.name for im in other_entries]
        if os.path.exists(test_txt):
            if gt_mode == "replace":
                existing = read_test_txt(test_txt)
                train_images_all = [item for item in train_images_all if item.name not in existing]
                train_max_id = train_max_id - len(existing)
                for img in existing:
                    img_path = os.path.join(train_images_dir, img)
                    if os.path.exists(img_path):
                        os.remove(img_path)

                write_test_txt(test_txt, new_test_names)
                print("[GT] test.txt replaced.")
            elif gt_mode == "extend":
                existing = read_test_txt(test_txt)
                # Extend, de-duplicate by name while preserving order
                seen: Set[str] = set(existing)
                combined = list(existing)
                for n in new_test_names:
                    if n not in seen:
                        combined.append(n)
                        seen.add(n)
                write_test_txt(test_txt, combined)
                print("[GT] test.txt extended.")
        else:
            write_test_txt(test_txt, new_test_names)
            print("[GT] test.txt created.")

    # Renumber and append
    next_id = train_max_id
    for im in other_entries:
        next_id += 1
        im.id = next_id
        # Keep camera_id as-is (see header notes)
        train_images_all.append(im)

    # Write merged images.txt
    write_images_txt(train_images_txt, train_images_all)

    print("Merge complete. New images.txt written.")


# ---------------------------------
# CLI / Orchestration
# ---------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Merge COLMAP datasets by pose-aware logic and filename collision handling.\n"
            "Scenarios:\n"
            "  1) --ground_truth_dir + --train_dirs TRAIN [TRAIN ...] → GT merged into each TRAIN\n"
            "  2) --train_dirs TRAIN_A TRAIN_B [TRAIN_C ...]           → merge all others into TRAIN_A (no test.txt)\n"
        )
    )
    parser.add_argument(
        "--ground_truth_dir",
        type=str,
        default=None,
        help="Path to ground-truth dataset to merge into train dataset(s). Optional if merging train+train.",
    )
    parser.add_argument(
        "--train_dirs",
        type=str,
        nargs="+",
        required=True,
        help=(
            "TRAIN directories. If --ground_truth_dir is provided, you may pass 1 or more. "
            "If not provided, pass 2 or more: the first is the base, others are merged into it."
        ),
    )
    parser.add_argument(
        "--gt_mode",
        type=str,
        choices=["error", "extend", "replace"],
        default="error",
        help=(
            "Behavior if test.txt already exists in TRAIN (GT previously assigned):\n"
            "  error   → abort (default)\n"
            "  extend  → append new GT names to test.txt (skip name duplicates)\n"
            "  replace → overwrite test.txt with new GT names\n"
        ),
    )
    parser.add_argument(
        "--pose_tol",
        type=float,
        default=1e-6,
        help=(
            "Tolerance for pose-based duplicate detection on qvec+tvec. "
            "Values are quantized by tol and compared."
        ),
    )
    parser.add_argument(
        "--duplicate_resolution",
        type=str,
        choices=["error", "remaining"],
        default="error",
        help=(
            "Behavior if test.txt already exists in TRAIN (GT previously assigned):\n"
            "  error      → abort (default)\n"
            "  remaining  → only add non-duplicated images\n"
        ),
    )

    args = parser.parse_args()

    if args.ground_truth_dir:
        # GT + one or more train dirs
        if len(args.train_dirs) < 1:
            raise SystemExit("Provide at least one --train_dirs when using --ground_truth_dir.")
        for train_dir in args.train_dirs:
            merge_into_train(
                train_dir=train_dir,
                other_dir=args.ground_truth_dir,
                is_gt=True,
                gt_mode=args.gt_mode,
                pose_tol=args.pose_tol,
                duplicate_resolution=args.duplicate_resolution,
            )
    else:
        # TRAIN + TRAIN (no GT). Need at least 2 train dirs; merge others into the first.
        if len(args.train_dirs) < 2:
            raise SystemExit(
                "Provide at least two --train_dirs when not using --ground_truth_dir. "
                "The first is the base; others are merged into it."
            )
        base = args.train_dirs[0]
        for other in args.train_dirs[1:]:
            merge_into_train(
                train_dir=base,
                other_dir=other,
                is_gt=False,
                gt_mode=args.gt_mode,  # ignored when is_gt=False
                pose_tol=args.pose_tol,
                duplicate_resolution=args.duplicate_resolution,
            )


if __name__ == "__main__":
    main()
