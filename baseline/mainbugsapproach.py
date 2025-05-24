import pandas as pd
import tempfile
import subprocess
import os
from radon.complexity import cc_visit
from radon.metrics import mi_visit

# === Update these paths to your file locations ===
embpp_file = "Embpp_Results.xlsx"
humaneval_file = "HumanEvalu_Results.xlsx"

# === Load datasets ===
df_embpp = pd.read_excel(embpp_file)
df_humaneval = pd.read_excel(humaneval_file)

# === Normalize column names ===
df_embpp.columns = df_embpp.columns.str.strip().str.lower()
df_humaneval.columns = df_humaneval.columns.str.strip().str.lower()

# === Function to introduce a simple bug ===
def introduce_bug(code):
    if not isinstance(code, str):
        return ""
    return code.replace("==", "!=").replace(">", "<")

# === Function to evaluate test case execution ===
def evaluate_test(code, test_case):
    bug, false_alarm = 0, 0
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "script.py")
            with open(path, "w", encoding="utf-8") as f:
                f.write(code + "\n\n" + test_case)
            result = subprocess.run(["python", path], capture_output=True, timeout=5)
            if result.returncode != 0:
                bug = 1
    except Exception:
        false_alarm = 1
    return bug, false_alarm

# === Function to compute maintainability and complexity ===
def analyze_code(code):
    try:
        mi = mi_visit(code, False)
        cc = sum(block.complexity for block in cc_visit(code))
        return round(mi, 2), cc
    except Exception:
        return 0.0, 0

# === Core function ===
def process_dataset(df, output_path):
    results = []
    for idx, row in df.iterrows():
        code = str(row.get("fixed code", ""))
        test_case = str(row.get("test case", ""))

        buggy_code = introduce_bug(code)
        bug, false_alarm = evaluate_test(buggy_code, test_case)
        mi, cc = analyze_code(buggy_code)

        results.append({
            "Index": idx,
            "Bug Introduced Code": buggy_code,
            "Test Case": test_case,
            "Detected Bug": bug,
            "False Alarm": false_alarm,
            "Maintainability Index": mi,
            "Cyclomatic Complexity": cc
        })

    pd.DataFrame(results).to_excel(output_path, index=False)
    print(f"✅ Results saved to {output_path}")

# === Run for both datasets ===
process_dataset(df_embpp, "EMBPP_Results.xlsx")
process_dataset(df_humaneval, "HumanEval_Results.xlsx")
