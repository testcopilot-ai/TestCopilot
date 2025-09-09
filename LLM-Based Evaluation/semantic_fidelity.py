#!/usr/bin/env python3
"""
Semantic Fidelity (Temp_0 only)

Input CSV can be in either format:

1) WIDE (one row per repaired test):
   dataset, temp, task_id, label_r1, label_r2
   e.g., HumanEval, Temp_0, 38, retained, minor_drift

2) LONG (two rows per repaired test: one per annotator):
   dataset, temp, task_id, annotator, label
   e.g., HumanEval, Temp_0, 38, r1, retained
         HumanEval, Temp_0, 38, r2, minor_drift

Usage:
    python semantic_fidelity_temp0.py --csv annotations.csv \
        --excluded-mbpp 151 --excluded-humaneval 163
"""

import argparse
import sys
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from sklearn.metrics import cohen_kappa_score

# ------------- config -------------
RET_LABELS = {"retained", "keep", "no_drift", "ok"}
MINOR_DRIFT = {"minor_drift", "minor", "slight"}
MAJOR_DRIFT = {"major_drift", "major", "discard", "diverged"}
TEMP_DEFAULT_KEYS = {"temp_0", "default", "0"}
N_BOOT = 10_000
SEED = 42
# ----------------------------------

def _norm(s: str) -> str:
    return str(s).strip().lower()

def normalize_label(s: str) -> str:
    x = _norm(s)
    if x in RET_LABELS:   return "retained"
    if x in MINOR_DRIFT:  return "minor_drift"
    if x in MAJOR_DRIFT:  return "major_drift"
    # fallback: try exact
    if x in {"retained", "minor_drift", "major_drift"}:
        return x
    raise ValueError(f"Unknown label: {s!r}")

def to_binary_retained(label: str) -> int:
    """1 if retained; 0 if drift (minor or major)."""
    return 1 if label == "retained" else 0

def bootstrap_ci_mean(x: np.ndarray, ci: float = 0.95, n_boot: int = N_BOOT, seed: int = SEED) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan, np.nan, np.nan
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boots = x[idx].mean(axis=1)
    alpha = 1 - ci
    lo, hi = np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])
    return float(x.mean()), float(lo), float(hi)

def read_annotations(path: str) -> pd.DataFrame:
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    return df

def is_wide(df: pd.DataFrame) -> bool:
    cols = set(c.lower() for c in df.columns)
    return ("label_r1" in cols) and ("label_r2" in cols)

def to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Convert long → wide if needed; ensure (dataset,temp,task_id,label_r1,label_r2)."""
    lower = {c.lower(): c for c in df.columns}

    # standardize required columns
    dataset_col = lower.get("dataset", None)
    temp_col    = lower.get("temp", lower.get("temperature", None))
    task_col    = lower.get("task_id", lower.get("id", None))
    annot_col   = lower.get("annotator", None)
    label_col   = lower.get("label", None)

    if all(col is not None for col in [dataset_col, temp_col, task_col, annot_col, label_col]):
        d = df[[dataset_col, temp_col, task_col, annot_col, label_col]].copy()
        d.columns = ["dataset","temp","task_id","annotator","label"]
        d["label"] = d["label"].map(normalize_label)
        # pivot to wide by annotator (expect exactly two annotators)
        wide = d.pivot_table(index=["dataset","temp","task_id"],
                             columns="annotator", values="label", aggfunc="first").reset_index()
        # try to detect r1/r2 naming; otherwise, take first two columns
        ann_cols = [c for c in wide.columns if c not in ["dataset","temp","task_id"]]
        if len(ann_cols) < 2:
            raise ValueError("Found fewer than two annotators in long format.")
        # map to label_r1, label_r2 deterministically
        ann_cols = sorted(ann_cols)
        wide = wide.rename(columns={ann_cols[0]: "label_r1", ann_cols[1]: "label_r2"})
        return wide[["dataset","temp","task_id","label_r1","label_r2"]]

    # Already wide?
    if is_wide(df):
        w = df.copy()
        # rename exactly
        rename_map = {}
        for c in df.columns:
            lc = c.lower()
            if lc == "dataset": rename_map[c] = "dataset"
            if lc in {"temp","temperature"}: rename_map[c] = "temp"
            if lc in {"task_id","id"}: rename_map[c] = "task_id"
            if lc == "label_r1": rename_map[c] = "label_r1"
            if lc == "label_r2": rename_map[c] = "label_r2"
        w = w.rename(columns=rename_map)
        return w[["dataset","temp","task_id","label_r1","label_r2"]]

    raise ValueError("CSV does not look like long or wide format. "
                     "Expected columns for long: dataset,temp,task_id,annotator,label "
                     "or wide: dataset,temp,task_id,label_r1,label_r2.")

def filter_temp0(wide: pd.DataFrame) -> pd.DataFrame:
    t = wide["temp"].astype(str).str.lower()
    mask = t.isin(TEMP_DEFAULT_KEYS) | t.str.contains(r"\btemp[_\s]*0\b")
    return wide[mask].copy()

def compute_stats(wide: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    # Normalize labels
    wide["label_r1"] = wide["label_r1"].map(normalize_label)
    wide["label_r2"] = wide["label_r2"].map(normalize_label)

    # Cohen's kappa on 3 classes
    kappa = cohen_kappa_score(wide["label_r1"], wide["label_r2"])

    # Consensus rule: retained if BOTH annotators say retained; otherwise drift
    retained_bin = (wide["label_r1"].eq("retained") & wide["label_r2"].eq("retained")).astype(int)

    mean_all, lo_all, hi_all = bootstrap_ci_mean(retained_bin.values)

    out = {
        "overall": {
            "n": int(len(wide)),
            "retained_mean": mean_all, "retained_ci_lo": lo_all, "retained_ci_hi": hi_all,
            "drift_mean": 1.0 - mean_all,
            "kappa": float(kappa),
        }
    }

    # by dataset
    for ds, sub in wide.groupby("dataset"):
        rb = (sub["label_r1"].eq("retained") & sub["label_r2"].eq("retained")).astype(int).values
        m, lo, hi = bootstrap_ci_mean(rb)
        out[ds] = {
            "n": int(len(sub)),
            "retained_mean": m, "retained_ci_lo": lo, "retained_ci_hi": hi,
            "drift_mean": 1.0 - m,
        }

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to annotations CSV/XLSX (Temp_0 and two reviewers).")
    ap.add_argument("--excluded-mbpp", type=int, default=None, help="Total MBPP excluded count (optional).")
    ap.add_argument("--excluded-humaneval", type=int, default=None, help="Total HumanEval excluded count (optional).")
    args = ap.parse_args()

    df = read_annotations(args.csv)
    wide = to_wide(df)
    wide = filter_temp0(wide)

    if wide.empty:
        print("No Temp_0 rows found. Check your 'temp' column values.", file=sys.stderr)
        sys.exit(1)

    stats = compute_stats(wide)

    # Pretty print
    def pct(x): 
        return f"{100*x:.1f}%"

    print("=== Semantic Fidelity (Temp_0 only) ===")
    print(f"Samples (overall): {stats['overall']['n']}")
    print(f"Retained: {pct(stats['overall']['retained_mean'])} "
          f"[{pct(stats['overall']['retained_ci_lo'])}, {pct(stats['overall']['retained_ci_hi'])}]")
    print(f"Drift: {pct(stats['overall']['drift_mean'])}")
    print(f"Cohen's kappa (3-class): {stats['overall']['kappa']:.3f}")

    for ds in sorted([k for k in stats.keys() if k not in {'overall'}]):
        s = stats[ds]
        print(f"\n[{ds}] n={s['n']}")
        print(f"  Retained: {pct(s['retained_mean'])} "
              f"[{pct(s['retained_ci_lo'])}, {pct(s['retained_ci_hi'])}]")
        print(f"  Drift: {pct(s['drift_mean'])}")

    if args.excluded_mbpp is not None:
        print(f"\nExcluded MBPP tests: {args.excluded_mbpp}")
    if args.excluded_humaneval is not None:
        print(f"Excluded HumanEval tests: {args.excluded_humaneval}")

    # Optional: write out a CSV with per-sample consensus
    wide_out = wide.copy()
    wide_out["consensus"] = np.where(
        wide_out["label_r1"].eq("retained") & wide_out["label_r2"].eq("retained"),
        "retained",
        "drift"
    )
    wide_out.to_csv("semantic_fidelity_temp0_samples.csv", index=False)
    print("\nWrote per-sample consensus to semantic_fidelity_temp0_samples.csv")

if __name__ == "__main__":
    main()
