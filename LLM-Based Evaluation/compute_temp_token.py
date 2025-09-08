#!/usr/bin/env python3
"""
Build 'Impact of Temperature, various Token Size on Pass Rates' table.

- Inputs: one file per configuration per dataset (or adjust the mapping below).
- Computes per-configuration Mean Pass Rate & Std Dev for HumanEval and MBPP.
- Runs pairwise t-tests between configurations:
    * Paired t-test if the same tasks are found in both configs (matched by task_id)
    * Welch's t-test otherwise
- Handles edge cases (all-perfect, zero variance) -> 'N/A'

Outputs:
  - out/temp_variation_table.tex  (LaTeX in your target format)
  - out/summary.csv               (numerical summary)
  - out/pairwise_tests.csv        (t / p per dataset)
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy.stats import ttest_rel, ttest_ind

# ========= USER CONFIG: point these to your files =========
DATA = {
    "HumanEval": {
        "Temp_0": "/mnt/data/HumanEvalu_Tools_AI_Evaluation_Results1.xlsx",
        "Temp_1": "/mnt/data/HumanEvalu_Tools_AI_Evaluation_Results_temp_1.xlsx",
        "Temp_2": "/mnt/data/HumanEvalu_Tools_AI_Evaluation_Results_temp_2.xlsx",
        "Temp_3": "/mnt/data/HumanEvalu_Tools_AI_Evaluation_Results_temp3.xlsx",
    },
    "MBPP": {
        "Temp_0": "/mnt/data/Mbpp_Tools_AI_Evaluation_Results1.xlsx",
        "Temp_1": "/mnt/data/Mbpp_Tools_AI_Evaluation_Results_temp_1.xlsx",
        "Temp_2": "/mnt/data/Mbpp_Tools_AI_Evaluation_Results_temp_2.xlsx",
        "Temp_3": "/mnt/data/Mbpp_Tools_AI_Evaluation_Results_temp3.xlsx",
    }
}
# Optional: human-readable labels to show in the table
CONFIG_LABELS = {
    "Temp_0": "Temp\\_0 (temp=default, MaxTok=default)",
    "Temp_1": "Temp\\_1 (Temp=0.3, MaxTok=800)",
    "Temp_2": "Temp\\_2 (Temp=0.5, MaxTok=500)",
    "Temp_3": "Temp\\_3 (Temp=0.7, MaxTok=700)",
}
# ==========================================================

OUTDIR = Path("out")
OUTDIR.mkdir(parents=True, exist_ok=True)

def _normkey(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def _read_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    if p.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(p)
    return pd.read_csv(p)

def _find_col(df: pd.DataFrame, candidates) -> Optional[str]:
    lut = {_normkey(c): c for c in df.columns}
    for c in candidates:
        k = _normkey(c)
        if k in lut:
            return lut[k]
    return None

def _get_task_id(df: pd.DataFrame) -> pd.Series:
    col = _find_col(df, ["task_id", "id", "problem_id", "case_id", "sample_id"])
    if col:
        return df[col].astype(str)
    # synthesize index if none present
    return pd.Series(np.arange(len(df)), index=df.index).astype(str)

def _coerce_pass(df: pd.DataFrame) -> pd.Series:
    """
    Try to infer a per-task pass indicator in [0,1].
    Priority:
      1) Boolean-ish Pass column
      2) 'Evaluation' exactly equals 1 (or 100 if percentages)
      3) 'Accuracy' or 'PassRate' columns (threshold > 0.999 -> pass)
    """
    # Boolean-ish pass
    pass_col = _find_col(df, ["Pass", "Passed", "is_pass", "TaskPass", "Outcome"])
    if pass_col is not None:
        s = df[pass_col]
        if s.dtype == bool:
            return s.astype(float)
        if pd.api.types.is_numeric_dtype(s):
            return (s.astype(float) > 0).astype(float)
        return s.astype(str).str.strip().str.lower().isin(
            ["1","true","t","yes","y"]
        ).astype(float)

    # Evaluation == 1 or 100
    eval_col = _find_col(df, ["Evaluation", "Eval", "EvalScore", "Eval_Score"])
    if eval_col is not None:
        s = pd.to_numeric(df[eval_col], errors="coerce")
        # if looks like percent:
        if s.dropna().mean() > 1.0:
            s = s / 100.0
        return (s >= 0.999).astype(float)

    # Accuracy-like
    acc_col = _find_col(df, ["Accuracy", "PassRate", "Pass_Rate"])
    if acc_col is not None:
        s = pd.to_numeric(df[acc_col], errors="coerce")
        if s.dropna().mean() > 1.0:
            s = s / 100.0
        return (s >= 0.999).astype(float)

    # Fallback: no detection -> all zeros
    return pd.Series(0.0, index=df.index)

def load_pass_vector(path: str) -> Tuple[pd.Series, pd.Series]:
    """Return (task_id, pass_float in {0.0,1.0})."""
    df = _read_any(path)
    tid = _get_task_id(df)
    p = _coerce_pass(df)
    # align indexes and return
    p = p.fillna(0.0).astype(float)
    return tid, p

def align_for_tests(tid_a: pd.Series, pass_a: pd.Series,
                    tid_b: pd.Series, pass_b: pd.Series) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Align vectors by task_id. If intersection non-empty, return paired arrays and paired=True.
    Else, return raw arrays and paired=False.
    """
    A = pd.DataFrame({"task_id": tid_a, "pa": pass_a})
    B = pd.DataFrame({"task_id": tid_b, "pb": pass_b})
    M = A.merge(B, on="task_id", how="inner")
    if len(M) >= 2:
        return M["pa"].values, M["pb"].values, True
    # fallback: unpaired
    return pass_a.values, pass_b.values, False

def mean_std(pass_vec: pd.Series) -> Tuple[float, float]:
    x = pass_vec.dropna().values
    return float(np.mean(x)), float(np.std(x, ddof=1)) if len(x) > 1 else 0.0

def safe_ttest(x: np.ndarray, y: np.ndarray, paired: bool) -> Tuple[str, str]:
    """
    Return (t_str, p_str) or ('N/A','N/A') if not meaningful:
      - length < 2
      - zero variance in both groups or identical vectors
    """
    if len(x) < 2 or len(y) < 2:
        return "N/A", "N/A"
    if (np.std(x, ddof=1) == 0 and np.std(y, ddof=1) == 0) or np.allclose(x, y):
        return "N/A", "N/A"
    try:
        if paired:
            t, p = ttest_rel(x, y, alternative="two-sided", nan_policy="omit")
        else:
            t, p = ttest_ind(x, y, equal_var=False, alternative="two-sided", nan_policy="omit")
        return f"{t:.4f}", f"{p:.4f}"
    except Exception:
        return "N/A", "N/A"

def build_summary_and_tests():
    # Order of configs for rows & comparisons
    configs = ["Temp_0", "Temp_1", "Temp_2", "Temp_3"]
    pairs = [("Temp_0","Temp_3"), ("Temp_0","Temp_1"), ("Temp_0","Temp_2"),
             ("Temp_1","Temp_2"), ("Temp_1","Temp_3"), ("Temp_2","Temp_3")]

    # Load per-config pass vectors
    store: Dict[str, Dict[str, Tuple[pd.Series, pd.Series]]] = {"HumanEval": {}, "MBPP": {}}
    for ds in ("HumanEval","MBPP"):
        for cfg in configs:
            tid, pv = load_pass_vector(DATA[ds][cfg])
            store[ds][cfg] = (tid, pv)

    # Build summary table
    rows = []
    for cfg in configs:
        he_mean, he_std = mean_std(store["HumanEval"][cfg][1])
        mb_mean, mb_std = mean_std(store["MBPP"][cfg][1])
        rows.append({
            "Configuration": CONFIG_LABELS.get(cfg, cfg),
            "HE_mean": he_mean, "HE_std": he_std,
            "MB_mean": mb_mean, "MB_std": mb_std
        })
    summary = pd.DataFrame(rows)

    # Overall summary (mean of means, std of means) – adapt to your desired definition
    overall = pd.DataFrame([{
        "Configuration": "Overall (All Configurations)",
        "HE_mean": summary["HE_mean"].mean(),
        "HE_std": summary["HE_std"].std(ddof=1) if len(summary)>1 else 0.0,
        "MB_mean": summary["MB_mean"].mean(),
        "MB_std": summary["MB_std"].std(ddof=1) if len(summary)>1 else 0.0,
    }])
    summary_with_overall = pd.concat([summary, overall], ignore_index=True)

    # Pairwise tests
    test_rows = []
    for a, b in pairs:
        # HumanEval
        ta, pa = store["HumanEval"][a]
        tb, pb = store["HumanEval"][b]
        x, y, paired = align_for_tests(ta, pa, tb, pb)
        t_he, p_he = safe_ttest(x, y, paired)

        # MBPP
        ta, pa = store["MBPP"][a]
        tb, pb = store["MBPP"][b]
        x, y, paired = align_for_tests(ta, pa, tb, pb)
        t_mb, p_mb = safe_ttest(x, y, paired)

        test_rows.append({
            "Pair": f"{a} vs {b}",
            "HE_t": t_he, "HE_p": p_he,
            "MB_t": t_mb, "MB_p": p_mb
        })
    tests = pd.DataFrame(test_rows)

    return summary_with_overall, tests

def to_latex_table(summary: pd.DataFrame, tests: pd.DataFrame) -> str:
    # Row order: Temp_0..Temp_3 then Overall
    order = ["Temp\\_0 (temp=default, MaxTok=default)",
             "Temp\\_1 (Temp=0.3, MaxTok=800)",
             "Temp\\_2 (Temp=0.5, MaxTok=500)",
             "Temp\\_3 (Temp=0.7, MaxTok=700)",
             "Overall (All Configurations)"]
    summary = summary.copy()
    # Ensure pretty rounding/format
    def fmt(x, is_std=False):
        if pd.isna(x):
            return "N/A"
        if is_std:
            # use ~ for approx if very small but nonzero
            return f"$\\approx${x:.3f}" if (x>0 and x<0.01) else f"{x:.4f}"
        return f"{x:.4f}"

    # Build header
    lines = []
    lines.append(r"\begin{table}")
    lines.append(r"\renewcommand{\arraystretch}{1.2}")
    lines.append(r"\caption{Impact of Temperature, various Token Size on Pass Rates}")
    lines.append(r"\label{tab:temp_variation_all}")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\begin{tabular}{llcccc}")
    lines.append(r"\toprule")
    lines.append(r"\multirow{2}{*}{\textbf{Configuration}} &  & \multicolumn{2}{c}{\textbf{HumanEval}} & \multicolumn{2}{c}{\textbf{MBPP}} \\")
    lines.append(r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}")
    lines.append(r"& & \textbf{Mean Pass Rate} & \textbf{Std Dev} & \textbf{Mean Pass Rate} & \textbf{Std Dev} \\")
    lines.append(r"\midrule")

    # Body rows
    for cfg in order[:-1]:
        row = summary[summary["Configuration"] == cfg]
        if row.empty:
            continue
        r = row.iloc[0]
        lines.append(
            f"{cfg} & & {fmt(r['HE_mean'])} & {fmt(r['HE_std'], True)} & {fmt(r['MB_mean'])} & {fmt(r['MB_std'], True)} \\\\"
        )
    # Overall
    row = summary[summary["Configuration"] == "Overall (All Configurations)"].iloc[0]
    lines.append(
        f"Overall (All Configurations) & & \\textbf{{{fmt(row['HE_mean'])}}} & \\textbf{{{fmt(row['HE_std'])}}} & \\textbf{{{fmt(row['MB_mean'])}}} & \\textbf{{{fmt(row['MB_std'])}}} \\\\"
    )
    lines.append(r"\midrule")
    lines.append(r"\textbf{Pairwise t-tests} & & \multicolumn{2}{c}{t / p-values (HumanEval)} & \multicolumn{2}{c}{t / p-values (MBPP)} \\")
    lines.append(r"\midrule")

    # Pairwise rows in the same order you wrote
    desired_pairs = ["Temp_0 vs Temp_3","Temp_0 vs Temp_1","Temp_0 vs Temp_2",
                     "Temp_1 vs Temp_2","Temp_1 vs Temp_3","Temp_2 vs Temp_3"]
    for pair in desired_pairs:
        r = tests[tests["Pair"] == pair]
        if r.empty:
            continue
        rr = r.iloc[0]
        he_cell = f"{rr['HE_t']} / {rr['HE_p']}" if rr['HE_t'] != "N/A" else "N/A & N/A"
        mb_cell = f"{rr['MB_t']} / {rr['MB_p']}" if rr['MB_t'] != "N/A" else "N/A & N/A"

        # For exact alignment with your sample, we put two cells: (t/p) then blanks or N/A placeholders.
        # You used: "& -0.9767 / 0.3294 & & N/A & N/A \\" for first row
        # We'll format as: "<pair> & & t/p (HE) & & t/p (MB) \\"
        lines.append(
            f"{pair.replace('_','\\_')} & & {rr['HE_t']} / {rr['HE_p']} & & {rr['MB_t']} / {rr['MB_p']} \\\\"
            if rr['HE_t'] != 'N/A' or rr['MB_t'] != 'N/A'
            else f"{pair.replace('_','\\_')} & & N/A & N/A & N/A & N/A \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)

def main():
    summary, tests = build_summary_and_tests()
    # Save CSVs
    summary.to_csv(OUTDIR/"summary.csv", index=False)
    tests.to_csv(OUTDIR/"pairwise_tests.csv", index=False)
    # Save LaTeX
    latex = to_latex_table(summary, tests)
    (OUTDIR/"temp_variation_table.tex").write_text(latex, encoding="utf-8")
    print("[OK] Wrote:", OUTDIR/"summary.csv")
    print("[OK] Wrote:", OUTDIR/"pairwise_tests.csv")
    print("[OK] Wrote:", OUTDIR/"temp_variation_table.tex")

if __name__ == "__main__":
    main()
