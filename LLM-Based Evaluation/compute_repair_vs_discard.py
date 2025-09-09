#!/usr/bin/env python3
"""
Compute per-configuration results for Repair vs Discard-Fail.

Outputs columns:
  dataset, temp, strategy, total_tests, bugs_detected, false_alarms

Usage:
  python compute_repair_discard.py /path/to/file_or_folder --out out.csv
"""

import argparse, re, sys
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd

# ---------- helpers ----------
def _normkey(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', str(s).lower())

def _read_any(p: Path) -> pd.DataFrame:
    if p.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    return pd.read_csv(p)

def _load_input(path: Path) -> pd.DataFrame:
    if path.is_file():
        return _read_any(path)
    if path.is_dir():
        frames = []
        for f in sorted(path.rglob("*")):
            if f.suffix.lower() in {".csv", ".xlsx", ".xls"}:
                try:
                    frames.append(_read_any(f).assign(_source=f.name))
                except Exception as e:
                    print(f"[WARN] skip {f}: {e}", file=sys.stderr)
        if not frames:
            raise FileNotFoundError("No CSV/XLSX files found.")
        return pd.concat(frames, ignore_index=True)
    raise FileNotFoundError(path)

def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lut = {_normkey(c): c for c in df.columns}
    for c in candidates:
        if _normkey(c) in lut:
            return lut[_normkey(c)]
    return None

def _coerce_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float) > 0
    return s.astype(str).str.strip().str.lower().isin(["1","true","t","yes","y"])

def _norm_strategy(x: str) -> str:
    s = str(x).strip().lower()
    if "repair" in s:
        return "Repair"
    if "discard" in s or "fail" in s or "drop" in s:
        return "Discard Fail"
    return str(x)

# ---------- core ----------
def compute_counts(df: pd.DataFrame, iteration_cap: int = 2) -> pd.DataFrame:
    # infer columns
    ds_col   = _find_col(df, ["dataset","bench","benchmark"])
    temp_col = _find_col(df, ["temp","temperature","config","setting"])
    strat_col= _find_col(df, ["strategy","mode","pipeline","method"])
    task_col = _find_col(df, ["task_id","id","problem_id","case_id","sample_id"])
    iter_col = _find_col(df, ["iteration","iter","step"])
    bug_col  = _find_col(df, ["bugdetected","bug_detected","detectedbug","is_bug","bug","foundbug","bugs"])
    fa_col   = _find_col(df, ["falsealarm","false_alarm","isfalsealarm","fa","falsealarms","numfalsealarms","fas"])

    missing = [n for n,c in {"dataset":ds_col,"temp":temp_col,"strategy":strat_col}.items() if c is None]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")

    # build normalized working columns
    d = df.copy()
    d["__dataset"]  = d[ds_col].astype(str)
    d["__temp"]     = d[temp_col].astype(str)
    d["__strategy"] = d[strat_col].astype(str).map(_norm_strategy)
    d["__task_id"]  = d[task_col] if task_col else np.arange(len(d))
    d["__iter"]     = pd.to_numeric(d[iter_col], errors="coerce").fillna(1).astype(int) if iter_col else 1
    d["__bug"]      = _coerce_bool(d[bug_col]) if bug_col else False
    d["__fa"]       = _coerce_bool(d[fa_col])  if fa_col  else False

    # keep first N iterations
    d = d[d["__iter"] <= iteration_cap]

    # aggregate to task level: any bug/fa within first 2 iterations
    task_agg = (
        d.groupby(["__dataset","__temp","__strategy","__task_id"], as_index=False)
         .agg(bug_any=("__bug","max"), fa_any=("__fa","max"))
    )

    # aggregate to configuration level
    conf = (
        task_agg.groupby(["__dataset","__temp","__strategy"], as_index=False)
                .agg(total_tests=("__task_id","nunique"),
                     bugs_detected=("bug_any","sum"),
                     false_alarms=("fa_any","sum"))
                .sort_values(["__dataset","__temp","__strategy"])
                .reset_index(drop=True)
    )

    # pretty names to match your table
    conf = conf.rename(columns={
        "__dataset":"dataset","__temp":"temp","__strategy":"strategy"
    })
    return conf

def print_like_table(conf: pd.DataFrame):
    # print in your layout (HumanEval then MBPP; Temp_0..Temp_3; Repair/Discard side by side)
    temp_order = ["Temp_0","Temp_1","Temp_2","Temp_3"]
    strat_order = ["Repair","Discard Fail"]

    for dataset in ["HumanEval","MBPP"]:
        sub = conf[conf["dataset"] == dataset]
        if sub.empty:
            continue
        print(f"\n=== {dataset} ===")
        for t in temp_order:
            row = sub[sub["temp"] == t]
            vals = {}
            for s in strat_order:
                r = row[row["strategy"] == s]
                if r.empty:
                    vals[s] = (0,0,0)
                else:
                    rr = r.iloc[0]
                    vals[s] = (int(rr["total_tests"]), int(rr["bugs_detected"]), int(rr["false_alarms"]))
            (tt_r, bd_r, fa_r) = vals["Repair"]
            (tt_d, bd_d, fa_d) = vals["Discard Fail"]
            print(f"{t:7s} | Total: {tt_r:3d}/{tt_d:3d} | Bugs: {bd_r:3d}/{bd_d:3d} | FAs: {fa_r:2d}/{fa_d:2d}")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="CSV/XLSX file OR folder containing result files")
    ap.add_argument("--iteration-cap", type=int, default=2, help="Max iteration to include (default=2)")
    ap.add_argument("--out", default="results.csv", help="Where to write the summary CSV")
    args = ap.parse_args()

    df = _load_input(Path(args.path))
    conf = compute_counts(df, iteration_cap=args.iteration_cap)
    conf.to_csv(args.out, index=False)
    print(f"[OK] wrote {args.out}")
    print_like_table(conf)

if __name__ == "__main__":
    main()
