# Easy as we already have everything in one json file
# Speed and gs count not available here


import argparse
import json
import math
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


    data = json.loads((root / "results.json").read_text())
    aggregated = defaultdict(lambda: defaultdict(list))
    out = []
    for dataset_name, dataset_data in data.items():
        
        # Adjust to fit
        # For GS
        #method = dataset_name.split("_")[-1]
        # For Ours / Unity
        method = "_".join(dataset_name.split("_")[2:])
        
        for metric in metrics:
            value = dataset_data.get(metric)
            aggregated[method][metric].append(value)
    
    for method, m_metrics in aggregated.items():
        row = {"method": method}
        for metric_n, metric_val in m_metrics.items():
            p10, p50, p90 = np_percentiles(metric_val)
            row[f"{metric_n}_p10"] = p10
            row[f"{metric_n}_p50"] = p50
            row[f"{metric_n}_p90"] = p90
        out.append(row)

    columns = ["method"] + [item for metric in metrics for item in [f"{metric}_p10", f"{metric}_p50", f"{metric}_p90"]]

    df = pd.DataFrame(out, columns=columns)
    df.to_csv(root / "aggregated_results.csv", index=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()

    aggregate(
        root=Path(args.root)
    )

