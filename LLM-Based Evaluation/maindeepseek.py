#!/usr/bin/env python3
"""
LLM-driven pipeline:
1) Local validation + optional Informant agent (PASS/FAIL).
2) If valid -> Fixer agent rewrites testcases to be runnable + aligned.
3) Prevents leakage: ONLY Buggy_Code is sent to LLM (never Correct Code).
4) Computes Maintainability Index (MI) locally for Buggy and Correct code.

Usage:
  python3 mainleetcodeagentai.py --input_csv /path/to/your.csv \
      --out_per_row agent_per_row.csv --out_summary agent_summary.csv --limit 50
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import time
from typing import Tuple, Dict, Any, List

import pandas as pd
import requests

try:
    from radon.metrics import mi_visit
except Exception:
    mi_visit = None


# # =========================================================
# # CONFIG
# # =========================================================
# CHATGPT_API_KEY = "YOUR_OPENAI_API_KEY"
# CHATGPT_BASE_URL = "https://api.openai.com/v1"
# CHATGPT_MODEL = "gpt-4-turbo"

# =========================================================
# CONFIG
# =========================================================
DEEPSEEK_API_KEY = "INSERT_YOUR_API"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-coder"


# =========================================================
# Helpers
# =========================================================
def s(x) -> str:
    """Safe stringify: converts NaN/None to empty string."""
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)


def get_col(row: pd.Series, *names: str) -> str:
    """Return the first non-empty value among candidate column names."""
    for n in names:
        v = s(row.get(n))
        if v.strip():
            return v
    return ""


def extract_method_name_from_fr(functional_requirement: str) -> str:
    """
    Robustly extract method name from FR text that contains a signature like:
    def checkIfPrerequisite(self, ...) -> List[bool]:
    """
    if not functional_requirement:
        return ""
    m = re.search(r"(?im)^\s*def\s+([A-Za-z_]\w*)\s*\(", functional_requirement)
    return m.group(1) if m else ""


def extract_method_name_from_code(code: str) -> str:
    if not code:
        return ""
    m = re.search(r"class\s+Solution\s*:\s*(?:.|\n)*?def\s+([A-Za-z_]\w*)\s*\(", code)
    if m:
        return m.group(1)
    m2 = re.search(r"def\s+([A-Za-z_]\w*)\s*\(", code)
    return m2.group(1) if m2 else ""


def fix_testcase_method_calls(testcase: str, method_name: str) -> str:
    """
    Replace sol.<anything>( -> sol.<method_name>(
    Replace self.solution.<anything>( -> self.solution.<method_name>(
    Replace Solution().<anything>( -> Solution().<method_name>(
    """
    if not isinstance(testcase, str) or not testcase.strip() or not method_name:
        return testcase

    testcase = re.sub(r"\bsol\s*\.\s*[A-Za-z_]\w*\s*\(", f"sol.{method_name}(", testcase)
    testcase = re.sub(
        r"\bself\s*\.\s*solution\s*\.\s*[A-Za-z_]\w*\s*\(",
        f"self.solution.{method_name}(",
        testcase,
    )
    testcase = re.sub(r"\bSolution\s*\(\s*\)\s*\.\s*[A-Za-z_]\w*\s*\(", f"Solution().{method_name}(", testcase)
    return testcase


def fix_pytest_parametrize_double_quotes(testcase: str) -> str:
    """
    Fix dataset bug:
      @pytest.mark.parametrize(""a,b,c"", [...])
    -> @pytest.mark.parametrize("a,b,c", [...])
    """
    if not testcase:
        return testcase
    return re.sub(
        r'@pytest\.mark\.parametrize\(\s*""(.*?)""\s*,',
        r'@pytest.mark.parametrize("\1",',
        testcase,
        flags=re.DOTALL,
    )


def is_python_syntax_ok(src: str) -> Tuple[bool, str]:
    """Local syntax check (no LLM)."""
    if not src or not src.strip():
        return False, "empty"
    try:
        ast.parse(src)
        return True, ""
    except Exception as e:
        return False, str(e)


def compute_mi(code: str) -> float:
    if mi_visit is None:
        return -1.0
    if not code or not isinstance(code, str):
        return -1.0
    try:
        return round(float(mi_visit(code, multi=True)), 2)
    except Exception:
        return -1.0


# =========================================================
# DeepSeek Call
# =========================================================
def chat_with_deepseek(user_input: str, model: str = DEEPSEEK_MODEL) -> str:
    if not DEEPSEEK_API_KEY or "REPLACE_WITH_YOUR_ROTATED_KEY" in DEEPSEEK_API_KEY:
        return "VERDICT: FAIL\nREASONS:\n- Missing API key."

    url = f"{DEEPSEEK_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a careful software engineering assistant."},
            {"role": "user", "content": user_input},
        ],
        "stream": False,
        "temperature": 0.1,
    }

    try:
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
        if r.status_code != 200:
            return f"VERDICT: FAIL\nREASONS:\n- HTTP {r.status_code}: {r.text[:800]}"
        data = r.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"VERDICT: FAIL\nREASONS:\n- Exception: {e}"


# =========================================================
# Informant Agent (OPTIONAL; now only checks basic consistency)
# =========================================================
def informant_agent(functional_requirement: str, scenario: str, testcase: str, buggy_code: str) -> Tuple[str, bool]:
    """
    Informant should NOT reject just because code is buggy.
    It should only reject for missing content / totally inconsistent format.
    """
    if not functional_requirement.strip():
        return "VERDICT: FAIL\nREASONS:\n- Missing functional_requirement.", False
    if not buggy_code.strip():
        return "VERDICT: FAIL\nREASONS:\n- Missing Buggy_Code.", False
    if not testcase.strip():
        return "VERDICT: FAIL\nREASONS:\n- Missing Testcases.", False

    prompt = f"""
Return EXACTLY:
VERDICT: PASS or FAIL
REASONS:
- ...

Only check:
- Fields exist (FR, Buggy_Code, Testcases)
- Testcases seems intended as pytest/python tests
- Buggy code seems intended as Python class Solution
Do NOT judge correctness.

Functional Requirement:
{functional_requirement}

Scenarios:
{scenario}

Testcases:
{testcase}

Buggy Code:
{buggy_code}
""".strip()

    response = chat_with_deepseek(prompt)
    verdict_pass = bool(re.search(r"VERDICT\s*:\s*PASS\b", response or "", re.I))
    return response, verdict_pass


# =========================================================
# Fixer Agent (returns ONLY fixed test file content)
# =========================================================
def fixer_agent(functional_requirement: str, scenario: str, testcase: str, buggy_code: str, method_name: str) -> str:
    prompt = f"""
You are a Python pytest test repair agent.

Task:
- Rewrite the GIVEN testcases into a runnable pytest test file.
- The fixed tests MUST call Solution().{method_name}(...) correctly.
- Keep it minimal and robust.
- Output ONLY the complete test file content (no markdown, no explanations).

Evaluate:
    - Evaluate test case effectiveness and bug detection.
    - Compute coverage: function, statement, path, etc.
    - Return % coverage and whether test case passes/fails.

Constraints:
- Use: from solution import Solution
- Allowed: import pytest
- No other non-stdlib imports
- No prints
- Prefer asserts

Functional Requirement:
{functional_requirement}

Scenarios:
{scenario}

Current Testcases:
{testcase}

Buggy Code (reference only; DO NOT change it):
{buggy_code}
""".strip()

    return chat_with_deepseek(prompt)


# =========================================================
# Main Pipeline
# =========================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True, help="Input CSV file")
    parser.add_argument("--out_per_row", default="agent_per_row.csv", help="Per-row output CSV")
    parser.add_argument("--out_summary", default="agent_summary.csv", help="Summary output CSV")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows to process")
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep between LLM calls (seconds)")
    parser.add_argument("--max_retries", type=int, default=2, help="Max retries for Fixer agent per row")
    parser.add_argument("--use_informant", action="store_true", help="Use LLM informant agent (optional)")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    # Normalize headers to handle Scenarios/Testcases/Code vs scenarios/testcases/code + truncations
    original_cols = list(df.columns)
    df.columns = [c.strip() for c in df.columns]

    if args.limit:
        df = df.head(args.limit)

    results: List[Dict[str, Any]] = []
    pass_count = 0
    fail_count = 0
    invalid_count = 0

    for idx, row in df.iterrows():
        dataset_id = row.get("Dataset_ID", row.get("Dataset_I", idx))

        # Robust column mapping (handles truncations)
        functional_requirement = get_col(row, "functional_requirement", "Functional_Requirement", "functional", "Function")
        scenario = get_col(row, "Scenarios", "scenarios", "Scenario")
        testcase = get_col(row, "Testcases", "testcases", "Testcase", "Ranked_Test_Case")
        buggy_code = get_col(row, "Buggy_Code", "Buggy_Co", "buggy_code", "buggy")
        correct_code = get_col(row, "Code", "code", "Correct_Code", "correct_code")

        # Infer method name from FR first, fallback to buggy code
        method_name = extract_method_name_from_fr(functional_requirement)
        if not method_name:
            method_name = extract_method_name_from_code(buggy_code)

        # Fix known dataset issues BEFORE validation
        testcase_fixed = fix_pytest_parametrize_double_quotes(testcase)
        testcase_fixed = fix_testcase_method_calls(testcase_fixed, method_name)

        # Local syntax gate (prevents arbitrary INVALID)
        ok_buggy, err_buggy = is_python_syntax_ok(buggy_code)
        ok_test, err_test = is_python_syntax_ok(testcase_fixed)

        if not ok_buggy or not ok_test:
            invalid_count += 1
            results.append({
                "Dataset_ID": dataset_id,
                "Status": "INVALID_SYNTAX",
                "Informant_Response": f"LocalSyntax: Buggy_OK={ok_buggy}({err_buggy}); Test_OK={ok_test}({err_test})",
                "Evaluation": "",
                "Fixed_Testcases": "",
                "Method_Inferred": method_name,
                "Leakage_Prevented": True,
                "Buggy_Code_Sent": False,   # not sent due to local gate
                "Correct_Code_Sent": False,
                "MI_Buggy": compute_mi(buggy_code),
                "MI_Correct": compute_mi(correct_code),
                "Testcases_Original": testcase,
                "Testcases_After_MethodFix": testcase_fixed,
            })
            continue

        # Optional informant (LLM)
        informant_response = "SKIPPED (local syntax ok)"
        is_valid = True
        if args.use_informant:
            informant_response, is_valid = informant_agent(functional_requirement, scenario, testcase_fixed, buggy_code)
            if not is_valid:
                invalid_count += 1
                results.append({
                    "Dataset_ID": dataset_id,
                    "Status": "INVALID",
                    "Informant_Response": informant_response,
                    "Evaluation": "",
                    "Fixed_Testcases": "",
                    "Method_Inferred": method_name,
                    "Leakage_Prevented": True,
                    "Buggy_Code_Sent": True,
                    "Correct_Code_Sent": False,
                    "MI_Buggy": compute_mi(buggy_code),
                    "MI_Correct": compute_mi(correct_code),
                    "Testcases_Original": testcase,
                    "Testcases_After_MethodFix": testcase_fixed,
                })
                continue

        # Fixer tries to produce fixed tests
        final_fixed = ""
        status = "FAIL"
        evaluation_text = ""

        for attempt in range(1, args.max_retries + 1):
            fixed = fixer_agent(functional_requirement, scenario, testcase_fixed, buggy_code, method_name)
            final_fixed = fixed

            # Quick heuristic: looks like pytest file
            looks_like_test = bool(re.search(r"(?m)^\s*def\s+test_", fixed or ""))
            has_solution_import = "from solution import Solution" in (fixed or "")
            ok_fixed, err_fixed = is_python_syntax_ok(fixed)

            if looks_like_test and has_solution_import and ok_fixed:
                status = "PASS"
                evaluation_text = f"Fixer produced syntactically valid pytest tests (attempt {attempt})."
                break

            evaluation_text = f"Fixer output not valid yet (attempt {attempt}): {err_fixed}"
            time.sleep(args.sleep)

        if status == "PASS":
            pass_count += 1
        else:
            fail_count += 1

        results.append({
            "Dataset_ID": dataset_id,
            "Status": status,
            "Informant_Response": informant_response,
            "Evaluation": evaluation_text,
            "Fixed_Testcases": final_fixed,
            "Method_Inferred": method_name,
            "Leakage_Prevented": True,
            "Buggy_Code_Sent": True,
            "Correct_Code_Sent": False,
            "MI_Buggy": compute_mi(buggy_code),
            "MI_Correct": compute_mi(correct_code),
            "Testcases_Original": testcase,
            "Testcases_After_MethodFix": testcase_fixed,
        })

        time.sleep(args.sleep)

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.out_per_row, index=False)

    summary = pd.DataFrame([{
        "Rows_Total": len(df),
        "Rows_Invalid": invalid_count,
        "Rows_Pass": pass_count,
        "Rows_Fail": fail_count,
        "Leakage_Prevented": True,
        "Model": DEEPSEEK_MODEL,
        "Columns_Seen": ", ".join([c.strip() for c in original_cols]),
    }])
    summary.to_csv(args.out_summary, index=False)

    print("DONE")
    print(summary.to_string(index=False))
    print(f"Per-row CSV: {args.out_per_row}")
    print(f"Summary CSV:  {args.out_summary}")


if __name__ == "__main__":
    main()
