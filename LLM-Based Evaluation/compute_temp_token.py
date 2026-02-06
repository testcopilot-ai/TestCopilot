#!/usr/bin/env python3
"""
Compute the impact of Temperature and Max Token Variation on Pass Rates.

- Loads results for HumanEval and MBPP across Temp_0..Temp_3.
- Computes mean pass rate & std dev per config.
- Runs pairwise t-tests between configurations.
- Outputs two CSVs:
    * summary.csv       (per-config mean/std)
    * pairwise_tests.csv (t / p-values for each config pair)
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from scipy.stats import ttest_rel, ttest_ind

# ========= USER CONFIG =========
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
OUTDIR = Path("out")
OUTDIR.mkdir(parents=True, exist_ok=True)
# ================================

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
        if _normkey(c) in lut:
            return lut[_normkey(c)]
    return None

def _get_task_id(df: pd.DataFrame) -> pd.Series:
    col = _find_col(df, ["task_id", "id", "problem_id", "case_id", "sample_id"])
    if col:
        return df[col].astype(str)
    return pd.Series(np.arange(len(df)), index=df.index).astype(str)

def _coerce_pass(df: pd.DataFrame) -> pd.Series:
    """Infer per-task pass indicator as {0.0,1.0}."""
    pass_col = _find_col(df, ["Pass", "Passed", "is_pass", "TaskPass", "Outcome"])
    if pass_col:
        s = df[pass_col]
        if s.dtype == bool:
            return s.astype(float)
        if pd.api.types.is_numeric_dtype(s):
            return (s.astype(float) > 0).astype(float)
        return s.astype(str).str.strip().str.lower().isin(
            ["1","true","t","yes","y"]
        ).astype(float)

    eval_col = _find_col(df, ["Evaluation","Eval","EvalScore"])
    if eval_col:
        s = pd.to_numeric(df[eval_col], errors="coerce")
        if s.dropna().mean() > 1.0:
            s = s / 100.0
        return (s >= 0.999).astype(float)

    acc_col = _find_col(df, ["Accuracy","PassRate"])
    if acc_col:
        s = pd.to_numeric(df[acc_col], errors="coerce")
        if s.dropna().mean() > 1.0:
            s = s / 100.0
        return (s >= 0.999).astype(float)

    return pd.Series(0.0, index=df.index)

def load_pass_vector(path: str) -> Tuple[pd.Series, pd.Series]:
    df = _read_any(path)
    tid = _get_task_id(df)
    p = _coerce_pass(df).fillna(0.0).astype(float)
    return tid, p

def align_for_tests(tid_a, pass_a, tid_b, pass_b) -> Tuple[np.ndarray, np.ndarray, bool]:
    A = pd.DataFrame({"task_id": tid_a, "pa": pass_a})
    B = pd.DataFrame({"task_id": tid_b, "pb": pass_b})
    M = A.merge(B, on="task_id", how="inner")
    if len(M) >= 2:
        return M["pa"].values, M["pb"].values, True
    return pass_a.values, pass_b.values, False

def mean_std(pass_vec: pd.Series) -> Tuple[float, float]:
    x = pass_vec.dropna().values
    return float(np.mean(x)), float(np.std(x, ddof=1)) if len(x) > 1 else 0.0

def safe_ttest(x, y, paired: bool) -> Tuple[str, str]:
    if len(x) < 2 or len(y) < 2:
        return "N/A","N/A"
    if (np.std(x, ddof=1)==0 and np.std(y, ddof=1)==0) or np.allclose(x,y):
        return "N/A","N/A"
    try:
        if paired:
            t,p = ttest_rel(x,y,nan_policy="omit")
        else:
            t,p = ttest_ind(x,y,equal_var=False,nan_policy="omit")
        return f"{t:.4f}", f"{p:.4f}"
    except Exception:
        return "N/A","N/A"

def main():
    configs = ["Temp_0","Temp_1","Temp_2","Temp_3"]
    pairs = [("Temp_0","Temp_3"),("Temp_0","Temp_1"),("Temp_0","Temp_2"),
             ("Temp_1","Temp_2"),("Temp_1","Temp_3"),("Temp_2","Temp_3")]

    store = {"HumanEval":{}, "MBPP":{}}
    for ds in ("HumanEval","MBPP"):
        for cfg in configs:
            tid,pv = load_pass_vector(DATA[ds][cfg])
            store[ds][cfg] = (tid,pv)

    # Summary
    rows = []
    for cfg in configs:
        he_mean,he_std = mean_std(store["HumanEval"][cfg][1])
        mb_mean,mb_std = mean_std(store["MBPP"][cfg][1])
        rows.append({"Configuration":cfg,"HE_mean":he_mean,"HE_std":he_std,
                     "MB_mean":mb_mean,"MB_std":mb_std})
    summary = pd.DataFrame(rows)

    # Pairwise tests
    tests = []
    for a,b in pairs:
        ta,pa = store["HumanEval"][a]
        tb,pb = store["HumanEval"][b]
        x,y,paired = align_for_tests(ta,pa,tb,pb)
        t_he,p_he = safe_ttest(x,y,paired)

        ta,pa = store["MBPP"][a]
        tb,pb = store["MBPP"][b]
        x,y,paired = align_for_tests(ta,pa,tb,pb)
        t_mb,p_mb = safe_ttest(x,y,paired)

        tests.append({"Pair":f"{a} vs {b}","HE_t":t_he,"HE_p":p_he,
                      "MB_t":t_mb,"MB_p":p_mb})
    tests = pd.DataFrame(tests)

    # Save
    summary.to_csv(OUTDIR/"summary.csv", index=False)
    tests.to_csv(OUTDIR/"pairwise_tests.csv", index=False)
    print("[OK] Wrote:", OUTDIR/"summary.csv")
    print("[OK] Wrote:", OUTDIR/"pairwise_tests.csv")

if __name__ == "__main__":
    main()
