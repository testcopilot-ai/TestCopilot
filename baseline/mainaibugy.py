import pandas as pd
import re
import traceback

# === File Paths ===
embpp_path = "mbpp-Agent_Tools_AI_Evaluation_Results.xlsx"
human_path = "HumanEvalu-Agent_Tools_AI_Evaluation_Results.xlsx"

# === Load Datasets ===
df_embpp = pd.read_excel(embpp_path)
df_human = pd.read_excel(human_path)

df_embpp.columns = df_embpp.columns.str.lower().str.strip()
df_human.columns = df_human.columns.str.lower().str.strip()

# === Diverse Mutation Generator ===
def generate_mutants(code: str):
    mutants = []

    op_swaps = {
        "==": "!=",
        "!=": "==",
        ">": "<",
        "<": ">",
        ">=": "<=",
        "<=": ">=",
        "+": "-",
        "-": "+",
        "*": "//",
        "//": "*",
        "True": "False",
        "False": "True"
    }
    for key, value in op_swaps.items():
        if key in code:
            mutated = code.replace(key, value, 1)
            mutants.append(mutated)

    const_patterns = [
        (r"\b0\b", "1"),
        (r"\b1\b", "0"),
        (r"\b-1\b", "1"),
        (r"\b10\b", "5"),
        (r"\b100\b", "50")
    ]
    for pattern, replacement in const_patterns:
        if re.search(pattern, code):
            mutated = re.sub(pattern, replacement, code, count=1)
            mutants.append(mutated)

    if "return " in code:
        mutated = re.sub(r"return\s+(.+)", r"return None", code, count=1)
        mutants.append(mutated)
        mutated = re.sub(r"return\s+(.+)", r"return 0", code, count=1)
        mutants.append(mutated)

    for cond in re.findall(r"if\s+([^\s:]+)", code):
        mutated = code.replace(f"if {cond}", f"if not {cond}", 1)
        mutants.append(mutated)

    structure_swaps = [("[]", "()"), ("()", "{}"), ("{}", "[]")]
    for old, new in structure_swaps:
        if old in code:
            mutated = code.replace(old, new, 1)
            mutants.append(mutated)

    return mutants

# === Convert print to assert ===
def convert_prints_to_asserts(test_code: str):
    lines = test_code.strip().splitlines()
    new_lines = []
    for line in lines:
        match = re.search(r'Expected: (.+?), Got: (.+?)\)?', line)
        if match:
            expected = match.group(1).strip()
            got_expr = match.group(2).strip()
            assert_line = f"assert {got_expr} == {expected}"
            new_lines.append(assert_line)
        else:
            new_lines.append(line)
    return "\n".join(new_lines)

# === Code Execution ===
def evaluate_code(code: str, test_code: str):
    try:
        cleaned_test = convert_prints_to_asserts(test_code)
        env = {}
        exec(code, env)
        exec(cleaned_test, env)
        return 0, ""
    except Exception as e:
        return 1, traceback.format_exc()

# === Evaluation Column Parser ===
def evaluation_flags(row):
    text = str(row.get("evaluation", "")).lower()
    keywords = ["fail", "fix", "incorrect", "issue", "error"]
    return int(any(kw in text for kw in keywords))

# === Main Analysis Function with Evaluation Support ===
def analyze(df, label="Dataset"):
    total = len(df)
    bugs = 0
    false_alarms = 0
    eval_bugs = 0
    logs = []

    for idx, row in df.iterrows():
        code = str(row.get("code", "")).strip()
        test_code = str(row.get("test case", "")).strip()

        if not code or not test_code:
            continue

        # Mutation-based bug detection
        mutants = generate_mutants(code)
        bug_found = False
        for m in mutants:
            fail, err = evaluate_code(m, test_code)
            if fail:
                bugs += 1
                bug_found = True
                logs.append({"dataset": label, "sample": idx, "type": "mutant", "error": err})
                break

        # False alarms on correct code
        fail_fixed, err_fixed = evaluate_code(code, test_code)
        if fail_fixed:
            false_alarms += 1
            logs.append({"dataset": label, "sample": idx, "type": "fixed", "error": err_fixed})

        # Evaluation-based bug detection
        if evaluation_flags(row):
            eval_bugs += 1

    return {
        "Total Samples": total,
        "Bugs Detected (via mutants)": bugs,
        "False Alarms (on fixed code)": false_alarms,
        "Evaluation-based Bugs": eval_bugs,
        "Error Logs": logs
    }

# === Run & Save ===
res_embpp = analyze(df_embpp, "EMBPP")
res_human = analyze(df_human, "HumanEval")

print("\n📊 EMBPP:", res_embpp)
print("\n📊 HumanEval:", res_human)

# === Save to Excel ===
pd.DataFrame([
    {k: v for k, v in res_embpp.items() if k != "Error Logs"},
    {k: v for k, v in res_human.items() if k != "Error Logs"}
], index=["EMBPP", "HumanEval"]).to_excel("bug_detection_summary.xlsx")

pd.DataFrame(res_embpp["Error Logs"] + res_human["Error Logs"]).to_excel("error_logs.xlsx", index=False)
print("\n✅ Saved to 'bug_detection_summary.xlsx' and 'error_logs.xlsx'")
