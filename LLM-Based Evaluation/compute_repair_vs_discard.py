#!/usr/bin/env python3
"""
Compute per-configuration results for Repair vs Discard-Fail:

For each (Dataset, Temp, Strategy) triple, compute:
- Total Tests
- Bugs Detected
- False Alarms

Inputs can be:
  1) A single merged CSV/XLSX with columns including dataset/temp/strategy, OR
  2) A directory containing multiple CSV/XLSX files, one or many per configuration.

Supports both:
  - Single-row-per-task results, and
  - Multi-iteration results (with columns like 'iteration' and 'task_id').
    In the multi-iteration case, we cap to two iterations and count a task as
    "Bug Detected" if ANY of its first two iterations detected a bug.
    False Alarms are counted analogously (ANY).
"""

import argparse, re, sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple

# --------------------------
# Helpers
# --------------------------

def normkey(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', str(s).lower())

def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lut = {normkey(c): c for c in df.columns}
    for c in candidates:
        if normkey(c) in lut:
            return lut[normkey(c)]
    return None

def coerce_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    if pd.api.types.is_numeric_dtype(s):
        return (s.astype(float) > 0)
    return s.astype(str).str.strip().str.lower().isin(["1","true","t","yes","y"])

def read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path)

def load_inputs(input_path: Path) -> pd.DataFrame:
    if input_path.is_file():
        return read_any(input_path)
    elif input_path.is_dir():
        frames = []
        for f in sorted(input_path.rglob("*")):
            if f.suffix.lower() in [".csv", ".xlsx", ".xls"]:
                try:
                    frames.append(read_any(f).assign(_source=str(f.name)))
                except Exception as e:
                    print(f"[WARN] Failed reading {f}: {e}", file=sys.stderr)
        if not frames:
            raise FileNotFoundError("No CSV/XLSX files found under directory.")
        return pd.concat(frames, ignore_index=True)
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

# --------------------------
# Core computation
# --------------------------

def compute_counts(df: pd.DataFrame, iteration_cap: int = 2) -> pd.DataFrame:
    """Compute Repair vs Discard-Fail totals per dataset/temp/strategy."""

    ds_col   = find_col(df, ["dataset","bench","benchmark"])
    temp_col = find_col(df, ["temp","temperature","config","setting"])
    strat_col= find_col(df, ["strategy","mode","pipeline","method"])
    task_col = find_col(df, ["task_id","id","problem_id","case_id","sample_id"])
    iter_col = find_col(df, ["iteration","iter","step"])
    bug_col  = find_col(df, ["bugdetected","is_bug","bug","foundbug","bugs"])
    fa_col   = find_col(df, ["falsealarm","fa","false_alarms","numfalsealarms"])

    if not (ds_col and temp_col and strat_col):
        raise ValueError("Missing one of dataset/temp/strategy columns in the input file.")

    bug_bool = coerce_bool(df[bug_col]) if bug_col else pd.Series(False, index=df.index)
    fa_bool  = coerce_bool(df[fa_col]) if fa_col else pd.Series(False, index=df.index)

    df = df.copy()
    df["__dataset"] = df[ds_col].astype(str)
    df["__temp"]    = df[temp_col].astype(str)
    df["__strategy"]= df[strat_col].astype(str).str.strip().str.title()
    df["__bug"]     = bug_bool.values
    df["__fa"]      = fa_bool.values
    df["__task_id"] = df[task_col] if task_col else np.arange(len(df))
    df["__iter"]    = pd.to_numeric(df[iter_col], errors="coerce").fillna(1).astype(int) if iter_col else 1

    # Cap iterations
    df_cap = df[df["__iter"] <= iteration_cap]

    # Aggregate to task-level
    agg = (
        df_cap.groupby(["__dataset","__temp","__strategy","__task_id"], as_index=False)
        .agg(bug_any=("__bug","max"), fa_any=("__fa","max"))
    )

    # Aggregate to config-level
    conf = (
        agg.groupby(["__dataset","__temp","__strategy"], as_index=False)
        .agg(
            total_tests=("__task_id","nunique"),
            bugs_detected=("bug_any","sum"),
            false_alarms=("fa_any","sum")
        )
    )
    return conf

def make_latex_table(conf: pd.DataFrame) -> str:
    temp_order = ["Temp_0","Temp_1","Temp_2","Temp_3"]
    strat_order = ["Repair","Discard Fail"]

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Per-configuration results for \textit{Repair} and \textit{Discard Fail}.}")
    lines.append(r"\label{tab:baseline-comparison-full}")
    lines.append(r"\renewcommand{\arraystretch}{0.7}")
    lines.append(r"\scriptsize")
    lines.append(r"\begin{adjustbox}{max width=\linewidth}")
    lines.append(r"\begin{tabular}{llcccccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Dataset} & \textbf{Config} & \multicolumn{2}{c}{\textbf{Total Tests}} & \multicolumn{2}{c}{\textbf{Bugs Detected}} & \multicolumn{2}{c}{\textbf{False Alarms}} \\")
    lines.append(r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}")
    lines.append(r"& & Repair & Discard Fail & Repair & Discard Fail & Repair & Discard Fail \\")
    lines.append(r"\midrule")

    for dataset in conf["__dataset"].unique():
        sub = conf[conf["__dataset"] == dataset]
        lines.append(rf"\multirow{{4}}{{*}}{{{dataset}}}")
        for i, temp in enumerate(temp_order):
            row = sub[sub["__temp"] == temp]
            vals = {}
            for strat in strat_order:
                r = row[row["__strategy"].str.contains(strat, case=False)]
                if r.empty:
                    vals[strat] = (0,0,0)
                else:
                    rr = r.iloc[0]
                    vals[strat] = (int(rr["total_tests"]), int(rr["bugs_detected"]), int(rr["false_alarms"]))
            t_r, b_r, f_r = vals["Repair"]
            t_d, b_d, f_d = vals["Discard Fail"]
            prefix = " & " if i>0 else " "
            lines.append(rf"{prefix}{temp} & {t_r} & {t_d} & {b_r} & {b_d} & {f_r} & {f_d} \\")
        lines.append(r"\midrule")

    if lines[-1] == r"\midrule":
        lines = lines[:-1]
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{adjustbox}")
    lines.append(r"\end{table}")

    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Path to CSV/XLSX file or directory of files.")
    ap.add_argument("--iteration-cap", type=int, default=2, help="Max iteration to consider (default=2).")
    ap.add_argument("--outdir", default=".", help="Output directory.")
    args = ap.parse_args()

    df = load_inputs(Path(args.input))
    conf = compute_counts(df, iteration_cap=args.iteration_cap)
    conf = conf.sort_values(["__dataset","__temp","__strategy"]).reset_index(drop=True)

    # Save results
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    conf.to_csv(outdir/"results.csv", index=False)
    with open(outdir/"table.tex","w",encoding="utf-8") as f:
        f.write(make_latex_table(conf))

    print(f"[OK] Saved: {outdir/'results.csv'}")
    print(f"[OK] Saved: {outdir/'table.tex'}")

if __name__ == "__main__":
    main()
