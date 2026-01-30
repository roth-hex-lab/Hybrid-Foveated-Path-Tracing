# - Reads per-trial metrics from
#     ROOT/<dataset>/train/<N>p_<S>s_<TRIAL>/results.json
# - Reads per-dataset render_speed.csv at
#     ROOT/<dataset>/render_speed.csv
# - Produces:
#     1) per_dataset_pxx.csv         -- per dataset/config/iteration: p10/p50/p90 across trials
#     2) overall_pxx_pooled.csv      -- pooled percentiles across ALL trials
#     3) overall_pxx_relative.csv    -- LPIPS/PSNR/SSIM/Masked_* with metric-prefixed names.
#
# Usage examples:
#   python aggregate_time_quality_relative.py --root ./eval/fast --method_prefix ours --train_cost_per_100_ms 120

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


CONFIG_RE = re.compile(r"(?P<n>\d+)p_(?P<s>\d+)s_(?P<t>\d+)$")
METHOD_ITER_RE = re.compile(r"^(?P<prefix>[A-Za-z0-9\-]+)_(?P<iter>\d+)$")

# Add masked variants; keep canonical names (columns use .lower())
METRICS = ["LPIPS", "PSNR", "SSIM", "Masked_PSNR", "Masked_SSIM", "GS_COUNT"]


def parse_config_name(name: str) -> Tuple[int, int, int]:
    m = CONFIG_RE.match(name)
    if not m:
        raise ValueError(f"Config directory '{name}' does not match '<N>p_<S>s_<TRIAL>'")
    return int(m.group("n")), int(m.group("s")), int(m.group("t"))


def load_render_speed_csv(dataset_dir: Path) -> Dict[Tuple[int, int], float]:
    csv_path = dataset_dir / "render_speed.csv"
    if not csv_path.exists():
        #raise FileNotFoundError(f"Missing render_speed.csv at '{csv_path}'")
        return {}
    
    df = pd.read_csv(csv_path)
    required = {"n_views", "samples", "best_avg_ms"}
    if not required.issubset(df.columns):
        raise ValueError(f"{csv_path} must contain columns {required}, got {set(df.columns)}")
    lookup = {}
    for _, row in df.iterrows():
        n = int(row["n_views"])
        s = int(row["samples"])
        best = float(row["best_avg_ms"])
        lookup[(n, s)] = best
    return lookup


def np_percentiles(values: List[float]) -> Tuple[float, float, float]:
    if not values:
        return (math.nan, math.nan, math.nan)
    arr = np.asarray(values, dtype=float)
    return (
        float(np.percentile(arr, 10)),
        float(np.percentile(arr, 50)),
        float(np.percentile(arr, 90)),
    )


def normalize_metric_dict(d: dict) -> Dict[str, float]:
    """
    Accepts any casing; picks only the metrics in METRICS (and ignores *_STDEV keys).
    """
    lowered = {k.lower(): v for k, v in d.items()}
    out = {}
    for m in METRICS:
        key = m.lower()
        if key in lowered:
            try:
                out[m] = float(lowered[key])
            except Exception:
                pass
    return out


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return math.nan


def aggregate(root: Path,
              method_prefix: str,
              train_cost_per_100_ms: float,
              pc_gen_overhead: float,
              per_dataset_csv: str,
              overall_pooled_csv: str,
              overall_relative_csv: str):

    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")

    datasets = [d for d in root.iterdir() if d.is_dir() and (d / "train").exists()]
    if not datasets:
        raise RuntimeError(f"No datasets with 'train' found under {root}")

    render_speed: Dict[str, Dict[Tuple[int, int], float]] = {}
    for ds_path in datasets:
        render_speed[ds_path.name] = load_render_speed_csv(ds_path)

    # trial_vals[(dataset, n, s, iter)][metric] = list of trial values
    trial_vals: Dict[Tuple[str, int, int, int], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    # Also capture time per trial (for pooled time stats)
    trial_times: Dict[Tuple[str, int, int, int], float] = {}

    for ds_path in datasets:
        train_dir = ds_path / "train"
        for cfg_dir in sorted(train_dir.iterdir()):
            if not cfg_dir.is_dir():
                continue
            try:
                n, s, t = parse_config_name(cfg_dir.name)
            except ValueError as e:
                print(f"[WARN] {e}")
                continue

            results_path = cfg_dir / "results.json"
            if not results_path.exists():
                print(f"[WARN] Missing results.json: {results_path}")
                continue

            #if (n, s) not in render_speed[ds_path.name]:
                #raise KeyError(f"{ds_path.name}/render_speed.csv lacks row for (n_views={n}, samples={s})")

            try:
                best_avg_ms = render_speed[ds_path.name][(n, s)]
            except:
                best_avg_ms = math.nan

            try:
                with open(results_path, "r") as f:
                    res = json.load(f)
            except Exception as e:
                print(f"[WARN] Failed to load JSON {results_path}: {e}")
                continue

            for method_key, metrics_dict in res.items():
                mm = METHOD_ITER_RE.match(method_key)
                if not mm:
                    continue
                if mm.group("prefix") != method_prefix:
                    continue
                iteration = int(mm.group("iter"))

                metrics = normalize_metric_dict(metrics_dict)
                # compute time for this trial/config/iteration
                time_ms = n * best_avg_ms + (iteration / 100.0) * train_cost_per_100_ms + pc_gen_overhead
                for m_name, m_val in metrics.items():
                    trial_vals[(ds_path.name, n, s, iteration)][m_name].append(float(m_val))
                trial_times[(ds_path.name, n, s, iteration)] = float(time_ms)

    if not trial_vals:
        raise RuntimeError("No trial metrics found. Check method prefix and folder structure.")

    # PER-DATASET aggregation: p10/p50/p90 across trials
    per_dataset_rows = []
    # pooled across all trials
    pooled_collect: Dict[Tuple[int, int, int], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    pooled_times: Dict[Tuple[int, int, int], List[float]] = defaultdict(list)

    for (ds, n, s, iteration), m_map in sorted(trial_vals.items(), key=lambda x: (x[0][1], x[0][2], x[0][3], x[0][0])):
        best_avg_ms = render_speed[ds][(n, s)]
        time_ms = trial_times.get((ds, n, s, iteration), n * best_avg_ms + (iteration / 100.0) * train_cost_per_100_ms + pc_gen_overhead)

        row = {
            "dataset": ds,
            "config": f"{n}pic{s}spp",
            "n_views": n,
            "samples": s,
            "iteration": iteration,
            "time_ms": float(time_ms),
        }

        total_trials = None
        # compute per-dataset percentiles for every metric
        for metric in METRICS:
            vals = m_map.get(metric, [])
            if total_trials is None:
                total_trials = len(vals)
            p10, p50, p90 = np_percentiles(vals)
            row[f"{metric.lower()}_p10"] = p10
            row[f"{metric.lower()}_p50"] = p50
            row[f"{metric.lower()}_p90"] = p90

            # pooled collects all trial values
            pooled_collect[(n, s, iteration)][metric].extend(vals)

        row["n_trials"] = int(total_trials or 0)
        per_dataset_rows.append(row)

        # pooled times: replicate time for each trial (weights pooled time like pooled values)
        if total_trials and total_trials > 0:
            pooled_times[(n, s, iteration)].extend([float(time_ms)] * total_trials)
        else:
            pooled_times[(n, s, iteration)].append(float(time_ms))

    # --- write per-dataset csv ---
    per_dataset_cols = [
        "dataset", "config", "n_views", "samples", "iteration", "time_ms",
        "lpips_p10", "lpips_p50", "lpips_p90",
        "psnr_p10", "psnr_p50", "psnr_p90",
        "ssim_p10", "ssim_p50", "ssim_p90",
        "masked_psnr_p10", "masked_psnr_p50", "masked_psnr_p90",
        "masked_ssim_p10", "masked_ssim_p50", "masked_ssim_p90",
        "gs_count_p10", "gs_count_p50", "gs_count_p90",
        "n_trials",
    ]
    pd.DataFrame(per_dataset_rows).sort_values(["dataset", "n_views", "samples", "iteration"]).to_csv(
        per_dataset_csv, index=False, columns=per_dataset_cols
    )

    # --- write pooled overall csv ---
    pooled_rows = []
    for (n, s, iteration), metric_map in sorted(pooled_collect.items()):
        out = {"config": f"{n}pic{s}spp", "n_views": n, "samples": s, "iteration": iteration}
        times = pooled_times[(n, s, iteration)]
        out["time_ms_mean"] = float(np.mean(times)) if times else math.nan
        out["time_ms_std"] = float(np.std(times, ddof=1)) if len(times) > 1 else 0.0
        t10, t50, t90 = np_percentiles(times) if times else (math.nan, math.nan, math.nan)
        out["time_ms_p10"], out["time_ms_p50"], out["time_ms_p90"] = t10, t50, t90


        present = next((len(v) for v in metric_map.values() if v), 0)
        out["n_trials_total"] = present

        for metric in METRICS:
            vals = metric_map.get(metric, [])
            p10, p50, p90 = np_percentiles(vals)
            out[f"{metric.lower()}_p10"] = p10
            out[f"{metric.lower()}_p50"] = p50
            out[f"{metric.lower()}_p90"] = p90
        pooled_rows.append(out)

    pooled_cols = [
        "config", "n_views", "samples", "iteration",
        "time_ms_mean", "time_ms_std", "time_ms_p10", "time_ms_p50", "time_ms_p90",
        "lpips_p10", "lpips_p50", "lpips_p90",
        "psnr_p10", "psnr_p50", "psnr_p90",
        "ssim_p10", "ssim_p50", "ssim_p90",
        "masked_psnr_p10", "masked_psnr_p50", "masked_psnr_p90",
        "masked_ssim_p10", "masked_ssim_p50", "masked_ssim_p90",
        "gs_count_p10", "gs_count_p50", "gs_count_p90",
        "n_trials_total",
    ]
    pd.DataFrame(pooled_rows).sort_values(["n_views", "samples", "iteration"]).to_csv(
        overall_pooled_csv, index=False, columns=pooled_cols
    )

    # --- Build lookup: per_dataset_lookup[(n,s,iteration)][dataset] = {metric: (p10,p50,p90)} ---
    per_dataset_lookup = defaultdict(dict)
    for r in per_dataset_rows:
        key = (int(r["n_views"]), int(r["samples"]), int(r["iteration"]))
        ds = r["dataset"]
        per_dataset_lookup[key][ds] = {}
        for metric in METRICS:
            per_dataset_lookup[key][ds][metric] = (
                safe_float(r.get(f"{metric.lower()}_p10")),
                safe_float(r.get(f"{metric.lower()}_p50")),
                safe_float(r.get(f"{metric.lower()}_p90")),
            )

    # --- Compute relative aggregates ---
    relative_rows = []
    for key in sorted(set(per_dataset_lookup.keys())):
        n, s, iteration = key
        ds_map = per_dataset_lookup[key]
        if not ds_map:
            continue

        out = {"config": f"{n}pic{s}spp", "n_views": n, "samples": s, "iteration": iteration}

        # time stats (pooled)
        times = pooled_times.get((n, s, iteration), [])
        out["time_ms_mean"] = float(np.mean(times)) if times else math.nan
        out["time_ms_std"]  = float(np.std(times, ddof=1)) if len(times) > 1 else 0.0
        p10t, p50t, p90t = np_percentiles(times) if times else (math.nan, math.nan, math.nan)
        out["time_ms_p10"], out["time_ms_p50"], out["time_ms_p90"] = p10t, p50t, p90t

        # For each metric, compute pooled p50 and relative deviation bands
        # Also store dataset_mean_*_p50 for each metric.
        for metric in METRICS:
            p50s_ds = []
            dev_low_ds = []
            dev_high_ds = []
            # collect per-dataset stats
            for ds, metrics_dict in ds_map.items():
                p10, p50, p90 = metrics_dict.get(metric, (math.nan, math.nan, math.nan))
                if math.isnan(p50):
                    continue
                p50s_ds.append(p50)
                dev_low_ds.append(p10 - p50)
                dev_high_ds.append(p90 - p50)

            # pooled p50 from all trials
            pooled_vals = pooled_collect.get((n, s, iteration), {}).get(metric, [])
            pooled_p10, pooled_p50, pooled_p90 = np_percentiles(pooled_vals)
            dataset_mean_p50 = float(np.mean(p50s_ds)) if len(p50s_ds) else math.nan

            if len(p50s_ds):
                mean_dev_low = float(np.mean(dev_low_ds))
                mean_dev_high = float(np.mean(dev_high_ds))
                max_dev_low = float(np.min(dev_low_ds))
                max_dev_high = float(np.max(dev_high_ds))
                avg_band_low = pooled_p50 + mean_dev_low
                avg_band_high = pooled_p50 + mean_dev_high
                max_band_low = pooled_p50 + max_dev_low
                max_band_high = pooled_p50 + max_dev_high
            else:
                mean_dev_low = mean_dev_high = max_dev_low = max_dev_high = math.nan
                avg_band_low = avg_band_high = max_band_low = max_band_high = math.nan

            mkey = metric.lower()

            out.update({
                f"pooled_{mkey}_p10": pooled_p10,
                f"pooled_{mkey}_p50": pooled_p50,
                f"pooled_{mkey}_p90": pooled_p90,
                f"dataset_mean_{mkey}_p50": dataset_mean_p50,
                f"{mkey}_mean_dev_low": mean_dev_low,
                f"{mkey}_mean_dev_high": mean_dev_high,
                f"{mkey}_max_dev_low": max_dev_low,
                f"{mkey}_max_dev_high": max_dev_high,
                f"{mkey}_avg_band_low": avg_band_low,
                f"{mkey}_avg_band_high": avg_band_high,
                f"{mkey}_max_band_low": max_band_low,
                f"{mkey}_max_band_high": max_band_high,
            })

        relative_rows.append(out)

    # --- write relative overall csv ---
    relative_cols = [
        "config", "n_views", "samples", "iteration",
        "time_ms_mean", "time_ms_std", "time_ms_p10", "time_ms_p50", "time_ms_p90",

        "pooled_lpips_p10", "pooled_lpips_p50", "pooled_lpips_p90",
        "dataset_mean_lpips_p50",
        "lpips_mean_dev_low", "lpips_mean_dev_high",
        "lpips_max_dev_low", "lpips_max_dev_high",
        "lpips_avg_band_low", "lpips_avg_band_high",
        "lpips_max_band_low", "lpips_max_band_high",
        
        "pooled_psnr_p10", "pooled_psnr_p50", "pooled_psnr_p90",
        "dataset_mean_psnr_p50",
        "psnr_mean_dev_low", "psnr_mean_dev_high",
        "psnr_max_dev_low", "psnr_max_dev_high",
        "psnr_avg_band_low", "psnr_avg_band_high",
        "psnr_max_band_low", "psnr_max_band_high",

        "pooled_ssim_p10", "pooled_ssim_p50", "pooled_ssim_p90",
        "dataset_mean_ssim_p50",
        "ssim_mean_dev_low", "ssim_mean_dev_high",
        "ssim_max_dev_low", "ssim_max_dev_high",
        "ssim_avg_band_low", "ssim_avg_band_high",
        "ssim_max_band_low", "ssim_max_band_high",

        "pooled_masked_psnr_p10", "pooled_masked_psnr_p50", "pooled_masked_psnr_p90",
        "dataset_mean_masked_psnr_p50",
        "masked_psnr_mean_dev_low", "masked_psnr_mean_dev_high",
        "masked_psnr_max_dev_low", "masked_psnr_max_dev_high",
        "masked_psnr_avg_band_low", "masked_psnr_avg_band_high",
        "masked_psnr_max_band_low", "masked_psnr_max_band_high",

        "pooled_masked_ssim_p10", "pooled_masked_ssim_p50", "pooled_masked_ssim_p90",
        "dataset_mean_masked_ssim_p50",
        "masked_ssim_mean_dev_low", "masked_ssim_mean_dev_high",
        "masked_ssim_max_dev_low", "masked_ssim_max_dev_high",
        "masked_ssim_avg_band_low", "masked_ssim_avg_band_high",
        "masked_ssim_max_band_low", "masked_ssim_max_band_high",
    ]
    pd.DataFrame(relative_rows).sort_values(["n_views", "samples", "iteration"]).to_csv(
        overall_relative_csv, index=False, columns=relative_cols
    )

    print(f"[OK] Wrote per-dataset CSV:        {per_dataset_csv}")
    print(f"[OK] Wrote pooled overall CSV:     {overall_pooled_csv}")
    print(f"[OK] Wrote relative overall CSV:   {overall_relative_csv}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--method_prefix", default="ours")
    ap.add_argument("--train_cost_per_100_ms", type=float, default=122.0, help="The training cost in milliseconds for 100 iterations of training. Value empirically averaged")
    ap.add_argument("--pc_gen_overhead", type=float, default=50, help="Pointcloud generation overhead in ms")
    ap.add_argument("--per_dataset_csv", default="per_dataset_pxx.csv")
    ap.add_argument("--overall_pooled_csv", default="overall_pxx_pooled.csv")
    ap.add_argument("--overall_relative_csv", default="overall_pxx_relative.csv")
    args = ap.parse_args()

    aggregate(
        root=Path(args.root),
        method_prefix=args.method_prefix,
        train_cost_per_100_ms=args.train_cost_per_100_ms,
        pc_gen_overhead=args.pc_gen_overhead,
        per_dataset_csv=args.per_dataset_csv,
        overall_pooled_csv=args.overall_pooled_csv,
        overall_relative_csv=args.overall_relative_csv,
    )

