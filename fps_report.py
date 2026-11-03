"""
Compute FPS statistics from the per-frame CSV logs written by main.py --save-csv.

Each CSV row is one processed frame with its timestamp (`ts`), so FPS can be
measured exactly instead of reading the 10-second "FPS snapshot" log lines.
The first `--warmup` seconds are skipped (TensorFlow/YOLO start-up is slow).

Usage:
    python fps_report.py fps_cpu.csv fps_gpu.csv
    python fps_report.py fps_cpu.csv fps_gpu.csv --warmup 10 --md
"""

import argparse
import os

import numpy as np
import pandas as pd


def stats(path, warmup):
    df = pd.read_csv(path)
    ts_all = df["ts"].to_numpy(dtype=float)
    keep = ts_all >= ts_all[0] + warmup
    df, ts = df[keep], ts_all[keep]
    if len(ts) < 2:
        raise SystemExit(f"{path}: not enough frames after warm-up")
    duration = ts[-1] - ts[0]
    # FPS in each whole second of the run
    sec = np.floor(ts - ts[0]).astype(int)
    per_sec = np.bincount(sec)[:-1] if sec[-1] > 0 else np.bincount(sec)
    frame_ms = np.diff(ts) * 1000
    no_face = (df["eye_source"].astype(str).isin(["none", "miss"]).mean()
               if "eye_source" in df else float("nan"))
    return {
        "run": os.path.splitext(os.path.basename(path))[0],
        "frames": len(ts),
        "seconds": duration,
        "mean_fps": (len(ts) - 1) / duration,
        "median_fps_per_sec": float(np.median(per_sec)),
        "min_fps_per_sec": float(per_sec.min()),
        "p95_frame_ms": float(np.percentile(frame_ms, 95)),
        "no_face_frames_%": 100 * no_face,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--warmup", type=float, default=10.0)
    ap.add_argument("--md", action="store_true", help="print a Markdown table for the README")
    args = ap.parse_args()

    rows = [stats(p, args.warmup) for p in args.csv]
    df = pd.DataFrame(rows).set_index("run")
    print(df.round(1).to_string())

    if args.md:
        print("\n| Run | Frames | Duration (s) | Mean FPS | Median FPS (1-s bins) | Min FPS (1-s bins) | 95th pct frame time (ms) |")
        print("|---|---|---|---|---|---|---|")
        for r in rows:
            print(f"| {r['run']} | {r['frames']} | {r['seconds']:.0f} | **{r['mean_fps']:.1f}** | "
                  f"{r['median_fps_per_sec']:.0f} | {r['min_fps_per_sec']:.0f} | {r['p95_frame_ms']:.0f} |")


if __name__ == "__main__":
    main()
