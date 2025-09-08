import os
import re
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("/mnt/data")

FILE_PATTERNS = {
    "HumanEval": [
        "HumanEvalu_Tools_AI_Evaluation_Results1.xlsx",        # Temp_0
        "HumanEvalu_Tools_AI_Evaluation_Results_temp_1.xlsx",  # Temp_1
        "HumanEvalu_Tools_AI_Evaluation_Results_temp_2.xlsx",  # Temp_2
        "HumanEvalu_Tools_AI_Evaluation_Results_temp3.xlsx",   # Temp_3
    ],
    "MBPP": [
        "Mbpp_Tools_AI_Evaluation_Results1.xlsx",              # Temp_0
        "Mbpp_Tools_AI_Evaluation_Results_temp_1.xlsx",        # Temp_1
        "Mbpp_Tools_AI_Evaluation_Results_temp_2.xlsx",        # Temp_2
        "Mbpp_Tools_AI_Evaluation_Results_temp3.xlsx",         # Temp_3
    ]
}
TEMP_LABELS = ["Temp_0", "Temp_1", "Temp_2", "Temp_3"]

# --- OPTIONAL: Override column names if auto-detection misses them ---
# Example:
# OVERRIDE = {
#   ("HumanEval","Temp_0"): {"eval_col": "Evaluation", "bug_col": "BugDetected", "fa_col": "FalseAlarm"},
#   ("MBPP","Temp_2"): {"eval_col": "Eval", "bug_col": "Bugs", "fa_col": "FAs"},
# }
OVERRIDE = {}

N_BOOT = 10_000
SEED = 42
rng = np.random.default_rng(SEED)

def bootstrap_mean_ci(x, n_boot=N_BOOT, ci=0.95):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan, (np.nan, np.nan)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = x[idx].mean(axis=1)
    alpha = 1 - ci
    lo, hi = np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])
    return float(np.nanmean(x)), (float(lo), float(hi))

def normkey(name): return re.sub(r'[^a-z0-9]', '', str(name).lower())

def find_col(df, cands):
    lut = {normkey(c): c for c in df.columns}
    for c in cands:
        k = normkey(c)
        if k in lut: return lut[k]
    return None

def coerce_eval(df, preferred=None):
    if preferred and preferred in df.columns:
        s = pd.to_numeric(df[preferred], errors="coerce")
    else:
        cands = ["Evaluation","Eval","Eval%","Eval_Percent","EvalPercent","PassRate","Accuracy"]
        col = find_col(df, cands)
        if not col: return None
        s = pd.to_numeric(df[col], errors="coerce")
    if s.dropna().mean() > 1.0: s = s / 100.0
    return s.clip(0,1)

def coerce_bool(s):
    if s.dtype == bool: return s
    if pd.api.types.is_numeric_dtype(s): return (s.astype(float) > 0)
    return s.astype(str).str.strip().str.lower().isin(["1","true","t","yes","y"])

def get_counts(df, bug_col=None, fa_col=None):
    bugs = fas = None
    if bug_col and bug_col in df.columns:
        s = df[bug_col]
        bugs = int(coerce_bool(s).sum()) if s.dtype != object else int(pd.to_numeric(s, errors="coerce").fillna(0).sum())
    else:
        bug_bool = ["BugDetected","bug_detected","DetectedBug","is_bug","Bug","FoundBug"]
        bug_num  = ["BugCount","Bugs","NumBugs","bugs_detected"]
        col = find_col(df, bug_bool)
        if col is not None:
            bugs = int(coerce_bool(df[col]).sum())
        else:
            col = find_col(df, bug_num)
            if col is not None:
                bugs = int(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())
    if fa_col and fa_col in df.columns:
        s = df[fa_col]
        fas = int(coerce_bool(s).sum()) if s.dtype != object else int(pd.to_numeric(s, errors="coerce").fillna(0).sum())
    else:
        fa_bool = ["FalseAlarm","false_alarm","IsFalseAlarm","FA"]
        fa_num  = ["FalseAlarms","NumFalseAlarms","FAs"]
        col = find_col(df, fa_bool)
        if col is not None:
            fas = int(coerce_bool(df[col]).sum())
        else:
            col = find_col(df, fa_num)
            if col is not None:
                fas = int(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())
    return bugs, fas

def fmt_bf(b,f):
    if b is None and f is None: return "N/A"
    if b is None: b="N/A"
    if f is None: f="N/A"
    return f"{b} / {f}"

def fmt_eval(m, lo, hi):
    if any(np.isnan([m,lo,hi])): return "N/A"
    return f"{m:.1f}\\% [{lo:.1f}, {hi:.1f}]"

results = {"HumanEval":{}, "MBPP":{}}

for dataset, files in FILE_PATTERNS.items():
    for i, fname in enumerate(files):
        temp = TEMP_LABELS[i]
        fpath = DATA_DIR / fname
        if not fpath.exists(): continue
        df = pd.read_excel(fpath)

        # Apply overrides if provided
        ov = OVERRIDE.get((dataset, temp), {})
        eval_col = ov.get("eval_col")
        bug_col  = ov.get("bug_col")
        fa_col   = ov.get("fa_col")

        s_eval = coerce_eval(df, preferred=eval_col)
        if s_eval is not None:
            mean, (lo, hi) = bootstrap_mean_ci(s_eval.dropna().values)
            mean, lo, hi = mean*100, lo*100, hi*100
        else:
            mean, lo, hi = (np.nan, np.nan, np.nan)
        bugs, fas = get_counts(df, bug_col=bug_col, fa_col=fa_col)

        results[dataset][temp] = {
            "bugs": bugs, "fas": fas, "eval_mean": mean, "eval_ci": (lo, hi)
        }

rows = []
for temp in TEMP_LABELS:
    he = results["HumanEval"].get(temp, {})
    mb = results["MBPP"].get(temp, {})
    he_bf = fmt_bf(he.get("bugs"), he.get("fas"))
    mb_bf = fmt_bf(mb.get("bugs"), mb.get("fas"))
    he_ev = fmt_eval(*(he.get("eval_mean", np.nan), *he.get("eval_ci", (np.nan, np.nan))))
    mb_ev = fmt_eval(*(mb.get("eval_mean", np.nan), *mb.get("eval_ci", (np.nan, np.nan))))
    rows.append((temp, he_bf, he_ev, mb_bf, mb_ev))

latex = r"""\begin{table}
\renewcommand{\arraystretch}{0.9}
\caption{Summary across temperatures: bug detections, false alarms, and evaluation robustness.}
\label{tab:rq1_concise}
\centering
\scriptsize
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{llcccc}
\toprule
\multirow{2}{*}{\textbf{Config}} & & \multicolumn{2}{c}{\textbf{HumanEval}} & \multicolumn{2}{c}{\textbf{MBPP}} \\
\cmidrule(lr){3-4} \cmidrule(lr){5-6}
& & \textbf{Bugs / FAs} & \textbf{Eval \% [95\% CI]} & \textbf{Bugs / FAs} & \textbf{Eval \% [95\% CI]} \\
\midrule
"""
for temp, he_bf, he_ev, mb_bf, mb_ev in rows:
    latex += f"{temp} & & {he_bf} & {he_ev} & {mb_bf} & {mb_ev} \\\\\n"
latex += r"""\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}
"""

out_path = DATA_DIR / "rq1_table.tex"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(latex)
print("LaTeX written to:", out_path)
