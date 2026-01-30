# Reads an overall CSV (either pooled or by-dataset) produced by the aggregator.
# Plots LPIPS vs time for each (n_views, samples) config. p50 is thick; p10 and p90 lighter.
# Optional: --shade to fill between p10 and p90.

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

COLOR_MAP = {} # actually happy with fallback colors
VS_FILTER = [(8, 16),(16, 4), (32, 4), (64,4), (64, 8), (64, 16)]

# Apply different line styles for sample counts
LINESTYLES = {4: "dotted", 8: "dashed", 16: "solid"}


def smooth_curve(x, y, points=200, k=3):
    """Smooth (x, y) using spline interpolation. k=3 cubic spline."""
    if len(x) < k+1:  # too few points, return original
        return x, y
    x_new = np.linspace(x.min(), x.max(), points)
    spline = make_interp_spline(x, y, k=k)
    y_new = spline(x_new)
    return x_new, y_new


def plot_time_quality(csv, out_file, args):
    df = pd.read_csv(csv)

    # strip whitespace from column names and string values.
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include=[object]).columns:
        # convert to str then strip; keeps NaNs as 'nan' only for pure-object columns,
        # then numeric coercion below will handle conversion to numbers.
        df[col] = df[col].astype(str).str.strip()

    # Coerce expected numeric columns to numeric types
    for col in ("n_views", "samples", "iteration", "time_ms_mean"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows missing grouping keys after coercion to avoid KeyError in groupby
    if "n_views" in df.columns and "samples" in df.columns:
        before = len(df)
        df = df.dropna(subset=["n_views", "samples"]) 
        after = len(df)
        if before != after:
            print(f"Dropped {before-after} rows with missing/invalid n_views or samples after coercion")

    # Convert grouping keys to integer type for consistent grouping
    if "n_views" in df.columns:
        df["n_views"] = df["n_views"].astype(int)
    if "samples" in df.columns:
        df["samples"] = df["samples"].astype(int)

    plt.figure(figsize=(15, 5))

    i = 0
    for (n, s), g in df.groupby(["n_views", "samples"], sort=True):
        if (n, s) in VS_FILTER:
            continue

        i += 0.001
        label = f"{n} views, {s} spp"
        g = g.sort_values("iteration")

        if "time_ms_mean" in g.keys():
            x = g["time_ms_mean"].values
        else:
            x = g["time_ms"].values

        color = COLOR_MAP.get(label, None)
        if color is None:
            # fallback: consistent color cycle if not in COLOR_MAP
            color = next(plt.gca()._get_lines.prop_cycler)["color"]

        linestyle = LINESTYLES[s]

        if args.center == 'pooled':
            y50 = g[f"pooled_{args.metric}_p50"].values
        elif args.center == 'dataset_mean':
            y50 = g[f"dataset_mean_{args.metric}_p50"].values
        else:
            raise ValueError("--center must be 'pooled' or 'dataset_mean'")
        
        y_avg_low, y_avg_high = g[f"{args.metric}_avg_band_low"].values, g[f"{args.metric}_avg_band_high"].values
        y_max_low, y_max_high = g[f"{args.metric}_max_band_low"].values, g[f"{args.metric}_max_band_high"].values

        x50, y50s = smooth_curve(x, y50)
        x_avg_low, y_avg_low = smooth_curve(x, y_avg_low)
        x_avg_high, y_avg_high = smooth_curve(x, y_avg_high)
        x_max_low, y_max_low = smooth_curve(x, y_max_low)
        x_max_high, y_max_high = smooth_curve(x, y_max_high)


        plt.scatter(x, y50, s=9, alpha=0.7, edgecolors=color, facecolors="none", linewidths=1.8, label=None)
        plt.plot(x50, y50s, linewidth=2.0, alpha=0.7, linestyle=linestyle, label=label, color=color)

        # Avg
        plt.plot(x_avg_low, y_avg_low, linewidth=1.1, alpha=0.2, linestyle=linestyle, color=color)
        plt.plot(x_avg_high, y_avg_high, linewidth=1.1, alpha=0.2, linestyle=linestyle, color=color)

        # Max
        plt.plot(x_max_low, y_max_low, linewidth=0.9, alpha=0.18, linestyle=linestyle, color=color)
        plt.plot(x_max_high, y_max_high, linewidth=0.9, alpha=0.18, linestyle=linestyle, color=color)

        if args.shade:
            plt.fill_between(x_avg_low, y_avg_low, y_avg_high, alpha=0.04, color=color)
            plt.fill_between(x_max_low, y_max_low, y_max_high, alpha=0.06, color=color)


        # Also plot initial generation duration
        rendertime_100it = x[1] - x[0]
        r_100_ms = x[0] - rendertime_100it
        rendertime_x = [r_100_ms, r_100_ms]
        rendertime_y = [0.1, 0.105] # ylim value
        plt.plot(rendertime_x, rendertime_y, linewidth=2.6, alpha=0.85, linestyle=linestyle, color=color)
        plt.scatter(rendertime_x[1], rendertime_y[1], s=9, alpha=0.85, edgecolors=color, facecolors=color, linewidths=1.8, label=None)


        #rendertime_x2 = [0, r_100_ms]
        #rendertime_y2 = [0.11 - i, 0.11 - i]
        #plt.plot(rendertime_x2, rendertime_y2, linewidth=1.5, alpha=0.7, linestyle=linestyle, color=color)




    if args.title:
        plt.title(args.title)
    plt.xlabel("Elapsed time (ms)", fontsize=12.5)
    plt.ylabel(args.metric.upper().replace("_", " ") + (" (lower is better)" if args.metric.lower() == "lpips" else ""), fontsize=12.5)
    plt.grid(True, linestyle=":", alpha=0.35)
    plt.legend(title="Parameters", fontsize=12.5, title_fontproperties={'weight':'bold'})
    plt.tick_params(colors="dimgray", labelsize=12)

    # Set x-axis to start at 0, highlight time taken to arrive at result
    plt.xlim((0, 1400))
    plt.ylim((0.1, 0.16))

    ax = plt.gca()
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("lightgray")

    ax.set_xticks(np.arange(0, 1400, 100))

    
    # Save as vector graphic (PDF for LaTeX)
    plt.savefig(out_file, bbox_inches="tight")
    print(f"Saved plot to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Aggregated CSV file")
    parser.add_argument("--metric", default="lpips", choices=["lpips", "psnr", "ssim", "masked_psnr", "masked_ssim"], help="Metric to plot")
    parser.add_argument("--out", default="graph.pdf", help="Output filename (use .pdf for LaTeX)")
    parser.add_argument("--shade", action="store_true", help="Shade p10-p90 interval")
    parser.add_argument("--center", default="pooled", choices=["pooled", "dataset_mean"], help="How center is calculated")
    parser.add_argument("--title", default="", help="Title")
    args = parser.parse_args()

    plot_time_quality(args.csv, args.out, args)

