"""
Find out why yawns are missed, from labelled webcam logs.

Record with:
    python main.py --device cpu --save-csv yawn_me.csv --show-mouth-crop
and press 'y' when a yawn starts and 'y' again when it ends.

Then:
    python analyze_yawn.py yawn_me.csv

The script reports, frame by frame:
  * how P(yawn) and MAR differ between labelled yawn and non-yawn frames,
  * how often the mouth was located by Haar vs. a fallback box,
  * recall / false-alarm rate of the current rule for several P(yawn) thresholds,
  * ROC AUC of P(yawn) and of MAR (1.0 = perfect separation, 0.5 = chance),
  * how many labelled yawns the program actually reported as "Yawn".

Thresholds chosen here are tuned on these recordings only. Check them on a
recording of a different person before trusting them.
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

MAR_THR = 0.34  # default in main.py


def load(paths, warmup):
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df = df[df["ts"] >= df["ts"].iloc[0] + warmup].copy()
        df["file"] = p
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    if "gt_yawn" not in df or df["gt_yawn"].sum() == 0:
        raise SystemExit("No labelled yawn frames: press 'y' at the start and end of each yawn.")
    return df


def segments(df):
    """Labelled yawn episodes as (file, start_idx, end_idx)."""
    out = []
    for f, g in df.groupby("file"):
        lab = g["gt_yawn"].to_numpy()
        idx = g.index.to_numpy()
        start = None
        for i, v in enumerate(lab):
            if v and start is None:
                start = i
            if (not v or i == len(lab) - 1) and start is not None:
                end = i if v else i - 1
                out.append((f, idx[start], idx[end]))
                start = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--warmup", type=float, default=5.0)
    args = ap.parse_args()

    df = load(args.csv, args.warmup)
    y = df["gt_yawn"].astype(int).to_numpy()
    p = df["mouth_prob"].to_numpy(float)
    mar = df["mar"].to_numpy(float)

    print(f"Frames: {len(df)}  (labelled yawn: {y.sum()}, not yawn: {(1 - y).sum()})\n")

    print("1) What the model sees")
    for name, m in [("yawn frames", y == 1), ("non-yawn frames", y == 0)]:
        print(f"   {name:16} mean P(yawn) = {np.nanmean(p[m]):.3f}   median = {np.nanmedian(p[m]):.3f}   "
              f"mean MAR = {np.nanmean(mar[m]):.3f}")
    if "mouth_source" in df:
        print("\n2) How the mouth box was found (share of frames)")
        print(pd.crosstab(df["mouth_source"], df["gt_yawn"].map({0: "not yawn", 1: "yawn"}),
                          normalize="columns").round(2).to_string())

    ok = ~np.isnan(p)
    print("\n3) Separation (ROC AUC; 1.0 = perfect, 0.5 = chance)")
    print(f"   P(yawn): {roc_auc_score(y[ok], p[ok]):.3f}")
    print(f"   MAR    : {roc_auc_score(y[ok], mar[ok]):.3f}")

    print(f"\n4) Frame rule 'MAR >= {MAR_THR} and P(yawn) >= thr' (Haar frames use MAR; others CNN only)")
    src = df["mouth_source"].astype(str).to_numpy() if "mouth_source" in df else np.array(["haar"] * len(df))
    gate = np.where(src == "haar", mar >= MAR_THR, True)
    print("   thr   recall(yawn)   false-alarm rate(non-yawn)")
    for thr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        pred = gate & (p >= thr)
        rec = pred[y == 1].mean()
        far = pred[y == 0].mean()
        print(f"   {thr:.1f}   {rec:12.2f}   {far:12.2f}")

    segs = segments(df)
    hit = sum((df.loc[s:e, "mouth_status"] == "Yawn").any() for _, s, e in segs)
    dur = [df.loc[e, "ts"] - df.loc[s, "ts"] for _, s, e in segs]
    print(f"\n5) Labelled yawns: {len(segs)}, reported as 'Yawn' by main.py: {hit}")
    print(f"   yawn durations (s): {', '.join(f'{d:.1f}' for d in dur)}")
    print("   (main.py needs the mouth open for > 1.4 s before it says 'Yawn')")


if __name__ == "__main__":
    main()
