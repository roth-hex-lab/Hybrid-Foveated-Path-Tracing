# Easy as we already have everything in one json file
# Speed and gs count not available here


import argparse
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

metrics = ["PSNR", "SSIM", "LPIPS", "Masked_PSNR", "Masked_SSIM", "FLIP_25PPD", "FLIP_50PPD"]

def np_percentiles(values: List[float]) -> Tuple[float, float, float]:
    if not values:
        return (math.nan, math.nan, math.nan)
    arr = np.asarray(values, dtype=float)
    return (
        float(np.percentile(arr, 10)),
        float(np.percentile(arr, 50)),
        float(np.percentile(arr, 90)),
    )

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return math.nan


def aggregate(root: Path):

    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")

    out = []
    aggregated = defaultdict(list)
    for dataset in os.listdir(root):
        if not (root / dataset).is_dir():
            continue
        path = (root / dataset / "render_speed.csv").resolve()
        data = pd.read_csv(path)

        for row in data.itertuples(index=False):
            aggregated[row.samples].append(row.best_avg_ms)

    for samples, times in aggregated.items():
        p10, p50, p90 = np_percentiles(times)
        out.append({
            "samples": samples,
            "time_p10": p10,
            "time_p50": p50,
            "time_p90": p90,
            "count": len(times)
        })

    df = pd.DataFrame(out, columns=["samples", "time_p10", "time_p50", "time_p90", "count"])
    df.to_csv(root / "aggregated_times.csv", index=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()

    aggregate(
        root=Path(args.root)
    )

