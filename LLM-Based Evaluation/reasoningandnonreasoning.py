#!/usr/bin/env python3
"""
LLM-driven pipeline (refined) + Reasoning ON/OFF experiment:

What this script does:
- Samples N rows per dataset (default 50) from an input CSV.
- For each sampled row:
  1) Local syntax validation (Buggy code + Testcases after light pre-fixes)
  2) Optional Informant agent (PASS/FAIL) [format-only check]
  3) Fixer agent generates a runnable pytest file
     - Runs in two modes:
       - Reasoning ON  => <REASONING> filled
       - Reasoning OFF => <REASONING> empty (forced)
- Prevents leakage: ONLY Buggy_Code is sent to LLM (never Correct code).
- Computes Maintainability Index (MI) locally for Buggy and Correct code (if radon installed).
- Stores reasoning separately without breaking pytest syntax.

Usage examples:
  # Run BOTH reasoning modes, sample 50 per dataset
  python3 mainleetcodeagentai.py --input_csv your.csv \
      --dataset_col Dataset --samples_per_dataset 50 \
      --out_per_row agent_per_row.csv --out_summary agent_summary.csv \
      --reasoning_mode both --use_informant

  # Only reasoning OFF
  python3 mainleetcodeagentai.py --input_csv your.csv \
      --dataset_col Dataset --samples_per_dataset 50 \
      --reasoning_mode off

Notes:
- This script checks syntax + "looks like pytest file" only.
- It does NOT execute tests against buggy/correct code here.
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
# CONFIG  (IMPORTANT: do NOT hardcode keys in real use)
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


def strip_code_fences(txt: str) -> str:
    """
    Removes markdown fences if present.
    Handles:
      ```python
      ...
      ```
    """
    if not txt or not isinstance(txt, str):
        return ""
    m = re.search(r"```(?:python)?\s*(.*?)\s*```", txt, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return txt.strip()


def fix_pytest_parametrize_double_quotes(testcase: str) -> str:
    """
    Fix dataset/LLM bug:
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


def ensure_single_solution_import(testcase: str) -> str:
    """Ensure only one 'from solution import Solution' line."""
    if not testcase:
        return testcase
    lines = testcase.splitlines()
    out = []
    seen = False
    for ln in lines:
        if ln.strip() == "from solution import Solution":
            if seen:
                continue
            seen = True
        out.append(ln)
    # if missing entirely, add at top
    if not seen:
        out.insert(0, "from solution import Solution")
    return "\n".join(out).strip() + "\n"


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


def extract_method_name_from_fr(functional_requirement: str) -> str:
    """Extract method name from FR signature like: def checkIfPrerequisite(self,..."""
    if not functional_requirement:
        return ""
    m = re.search(r"(?im)^\s*def\s+([A-Za-z_]\w*)\s*\(", functional_requirement)
    return m.group(1) if m else ""


def extract_method_name_from_code(code: str) -> str:
    """Fallback: extract first method name in class Solution."""
    if not code:
        return ""
    m = re.search(r"class\s+Solution\s*:\s*(?:.|\n)*?def\s+([A-Za-z_]\w*)\s*\(", code)
    if m:
        return m.group(1)
    m2 = re.search(r"def\s+([A-Za-z_]\w*)\s*\(", code)
    return m2.group(1) if m2 else ""


def fix_testcase_method_calls(testcase: str, method_name: str) -> str:
    """
    Force calls to Solution().{method_name}(...)
    Replace sol.<anything>( -> sol.<method_name>(
    Replace Solution().<anything>( -> Solution().<method_name>(
    """
    if not isinstance(testcase, str) or not testcase.strip() or not method_name:
        return testcase

    testcase = re.sub(r"\bsol\s*\.\s*[A-Za-z_]\w*\s*\(", f"sol.{method_name}(", testcase)
    testcase = re.sub(
        r"\bSolution\s*\(\s*\)\s*\.\s*[A-Za-z_]\w*\s*\(",
        f"Solution().{method_name}(",
        testcase,
    )
    return testcase


# =========================================================
# DeepSeek Call
# =========================================================
def chat_with_deepseek(user_input: str, model: str = DEEPSEEK_MODEL) -> str:
    if not DEEPSEEK_API_KEY or "INSERT_YOUR_API" in DEEPSEEK_API_KEY:
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
# LLM output parsing (Reasoning + Test file)
# =========================================================
REASONING_OPEN = "<REASONING>"
REASONING_CLOSE = "</REASONING>"
TEST_OPEN = "<TESTFILE>"
TEST_CLOSE = "</TESTFILE>"


def parse_reasoning_and_testfile(llm_text: str) -> Tuple[str, str]:
    """
    We ask the LLM to return:
      <REASONING>...</REASONING>
      <TESTFILE>...</TESTFILE>

    If it doesn't, we fallback:
      - reasoning=""
      - testfile = stripped content
    """
    if not llm_text:
        return "", ""

    t = llm_text.strip()

    # Try tagged parse
    if REASONING_OPEN in t and TEST_OPEN in t:
        r_match = re.search(
            rf"{re.escape(REASONING_OPEN)}(.*?){re.escape(REASONING_CLOSE)}",
            t,
            flags=re.DOTALL,
        )
        f_match = re.search(
            rf"{re.escape(TEST_OPEN)}(.*?){re.escape(TEST_CLOSE)}",
            t,
            flags=re.DOTALL,
        )
        reasoning = (r_match.group(1).strip() if r_match else "")
        testfile = (f_match.group(1).strip() if f_match else "")
        return reasoning, testfile

    # Fallback: remove fences only
    return "", strip_code_fences(t)


def postprocess_testfile(raw_test: str, method_name: str) -> str:
    """
    Hardening step AFTER the LLM:
    - strip fences
    - fix parametrize "".."" bug
    - ensure single import
    - fix method calls
    """
    t = strip_code_fences(raw_test)
    t = fix_pytest_parametrize_double_quotes(t)
    t = ensure_single_solution_import(t)
    t = fix_testcase_method_calls(t, method_name)
    return t.strip() + "\n"


def looks_like_pytest_file(test_src: str) -> bool:
    if not test_src or not test_src.strip():
        return False
    if "from solution import Solution" not in test_src:
        return False
    # must have at least one test_ function
    if not re.search(r"(?m)^\s*def\s+test_", test_src):
        return False
    return True


# =========================================================
# Informant Agent (OPTIONAL)
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
- Testcases looks like pytest/python tests
- Buggy code looks like Python class Solution
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
# Fixer Agent (Reasoning + Test file)
# =========================================================
def fixer_agent(
    functional_requirement: str,
    scenario: str,
    testcase: str,
    buggy_code: str,
    method_name: str,
    reasoning_on: bool = True,
) -> str:
    if reasoning_on:
        reasoning_rule = """
Return EXACTLY in this format:
<REASONING>
...brief reasoning (max 12 lines)...
</REASONING>
<TESTFILE>
...complete runnable pytest file...
</TESTFILE>
""".strip()
    else:
        reasoning_rule = """
Return EXACTLY in this format (REASONING must be EMPTY):
<REASONING>
</REASONING>
<TESTFILE>
...complete runnable pytest file...
</TESTFILE>
""".strip()

    prompt = f"""
You are a Python pytest test repair agent.

{reasoning_rule}

Rules for <TESTFILE>:
- Output ONLY python code (no markdown fences).
- Must contain: from solution import Solution
- Allowed: import pytest
- No other non-stdlib imports
- No prints
- Prefer asserts
- Tests MUST call: Solution().{method_name}(...)

Important:
- Do NOT invent random expected outputs.
- If multiple outputs are valid, assert invariants instead of exact equality.
- Use stable assertions (avoid ordering assumptions unless guaranteed).
- Make sure tests do NOT fail on correct code (FAR = 0).
- Optimize primarily for: Coverage, TCE, DDP, FAR.

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
    parser.add_argument("--out_per_row", default="agent_per_row.csv", help="Per-row output CSV (base name)")
    parser.add_argument("--out_summary", default="agent_summary.csv", help="Summary output CSV")
    parser.add_argument("--limit", type=int, default=None, help="Optional global limit AFTER sampling")
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep between LLM calls (seconds)")
    parser.add_argument("--max_retries", type=int, default=2, help="Max retries for Fixer agent per row")
    parser.add_argument("--use_informant", action="store_true", help="Use LLM informant agent (optional)")

    # --- New for ON/OFF experiment ---
    parser.add_argument("--dataset_col", default="Dataset", help="Column name containing dataset label/name")
    parser.add_argument("--samples_per_dataset", type=int, default=50, help="How many samples per dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--reasoning_mode", choices=["on", "off", "both"], default="both",
                        help="Run fixer with reasoning on/off/both")

    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    original_cols = list(df.columns)
    df.columns = [c.strip() for c in df.columns]

    # Sampling per dataset
    dataset_col = args.dataset_col.strip()
    if dataset_col not in df.columns:
        df["_DATASET_FALLBACK_"] = "ALL"
        dataset_col = "_DATASET_FALLBACK_"

    def sample_group(g: pd.DataFrame) -> pd.DataFrame:
        n = min(len(g), args.samples_per_dataset)
        return g.sample(n=n, random_state=args.seed)

    df = df.groupby(dataset_col, group_keys=False).apply(sample_group).reset_index(drop=True)

    # Optional global limit after sampling
    if args.limit:
        df = df.head(args.limit)

    # Determine which modes to run
    modes: List[Tuple[str, bool]] = []
    if args.reasoning_mode in ("on", "both"):
        modes.append(("ON", True))
    if args.reasoning_mode in ("off", "both"):
        modes.append(("OFF", False))

    results: List[Dict[str, Any]] = []

    # Counters by mode
    counters = {m[0]: {"PASS": 0, "FAIL": 0, "INVALID": 0} for m in modes}
    total_by_dataset_mode: Dict[Tuple[str, str], int] = {}

    for idx, row in df.iterrows():
        dataset_id = row.get("Dataset_ID", row.get("Dataset_I", idx))
        dataset_name = s(row.get(dataset_col)).strip() or "ALL"

        # Column mapping (your current columns)
        functional_requirement = get_col(row, "functional_requirement", "Functional_Requirement", "functional", "Function")
        scenario = get_col(row, "Scenarios", "scenarios", "Scenario")
        testcase = get_col(row, "Testcases", "testcases", "Testcase", "Ranked_Test_Case", "Fixed_Testcases")
        buggy_code = get_col(row, "Buggy_Code", "Buggy_Co", "buggy_code", "buggy")
        correct_code = get_col(row, "Code", "code", "Correct_Code", "correct_code")

        # Infer method name
        method_name = extract_method_name_from_fr(functional_requirement) or extract_method_name_from_code(buggy_code)

        # Pre-fix known dataset issues BEFORE LLM
        testcase_pre = strip_code_fences(testcase)
        testcase_pre = fix_pytest_parametrize_double_quotes(testcase_pre)
        testcase_pre = fix_testcase_method_calls(testcase_pre, method_name)
        testcase_pre = ensure_single_solution_import(testcase_pre)

        # Local syntax gate for buggy code + testcase (pre)
        ok_buggy, err_buggy = is_python_syntax_ok(buggy_code)
        ok_test, err_test = is_python_syntax_ok(testcase_pre)

        # Precompute MI once (same for ON/OFF)
        mi_buggy = compute_mi(buggy_code)
        mi_correct = compute_mi(correct_code)

        # Optional informant once (same for ON/OFF)
        informant_response = "SKIPPED (local syntax ok)"
        is_valid = True

        if not ok_buggy or not ok_test:
            # Invalid for ALL modes
            for mode_name, _ in modes:
                counters[mode_name]["INVALID"] += 1
                total_by_dataset_mode[(dataset_name, mode_name)] = total_by_dataset_mode.get((dataset_name, mode_name), 0) + 1

                results.append({
                    "Dataset": dataset_name,
                    "Dataset_ID": dataset_id,
                    "Mode": mode_name,
                    "Status": "INVALID_SYNTAX",
                    "Informant_Response": f"LocalSyntax: Buggy_OK={ok_buggy}({err_buggy}); Test_OK={ok_test}({err_test})",
                    "Evaluation": "",
                    "Fixed_Testcases": "",
                    "LLM_Reasoning": "",
                    "Method_Inferred": method_name,
                    "Leakage_Prevented": True,
                    "Buggy_Code_Sent": False,
                    "Correct_Code_Sent": False,
                    "MI_Buggy": mi_buggy,
                    "MI_Correct": mi_correct,
                    "Testcases_Original": testcase,
                    "Testcases_After_MethodFix": testcase_pre,
                    "Code": correct_code,
                    "Buggy_Code": buggy_code,
                })
            continue

        if args.use_informant:
            informant_response, is_valid = informant_agent(functional_requirement, scenario, testcase_pre, buggy_code)

        if args.use_informant and not is_valid:
            # Invalid for ALL modes
            for mode_name, _ in modes:
                counters[mode_name]["INVALID"] += 1
                total_by_dataset_mode[(dataset_name, mode_name)] = total_by_dataset_mode.get((dataset_name, mode_name), 0) + 1

                results.append({
                    "Dataset": dataset_name,
                    "Dataset_ID": dataset_id,
                    "Mode": mode_name,
                    "Status": "INVALID",
                    "Informant_Response": informant_response,
                    "Evaluation": "",
                    "Fixed_Testcases": "",
                    "LLM_Reasoning": "",
                    "Method_Inferred": method_name,
                    "Leakage_Prevented": True,
                    "Buggy_Code_Sent": True,
                    "Correct_Code_Sent": False,
                    "MI_Buggy": mi_buggy,
                    "MI_Correct": mi_correct,
                    "Testcases_Original": testcase,
                    "Testcases_After_MethodFix": testcase_pre,
                    "Code": correct_code,
                    "Buggy_Code": buggy_code,
                })
            continue

        # Run Fixer for each mode
        for mode_name, reasoning_on in modes:
            total_by_dataset_mode[(dataset_name, mode_name)] = total_by_dataset_mode.get((dataset_name, mode_name), 0) + 1

            final_fixed = ""
            final_reasoning = ""
            status = "FAIL"
            evaluation_text = ""

            for attempt in range(1, args.max_retries + 1):
                llm_out = fixer_agent(
                    functional_requirement=functional_requirement,
                    scenario=scenario,
                    testcase=testcase_pre,
                    buggy_code=buggy_code,
                    method_name=method_name,
                    reasoning_on=reasoning_on,
                )
                reasoning, testfile_raw = parse_reasoning_and_testfile(llm_out)

                final_reasoning = (reasoning or "").strip()
                final_fixed = postprocess_testfile(testfile_raw, method_name)

                ok_fixed, err_fixed = is_python_syntax_ok(final_fixed)
                if looks_like_pytest_file(final_fixed) and ok_fixed:
                    status = "PASS"
                    evaluation_text = f"[{mode_name}] Fixer produced syntactically valid pytest tests (attempt {attempt})."
                    break

                evaluation_text = f"[{mode_name}] Fixer output not valid yet (attempt {attempt}): {err_fixed}"
                time.sleep(args.sleep)

            if status == "PASS":
                counters[mode_name]["PASS"] += 1
            else:
                counters[mode_name]["FAIL"] += 1

            results.append({
                "Dataset": dataset_name,
                "Dataset_ID": dataset_id,
                "Mode": mode_name,
                "Status": status,
                "Informant_Response": informant_response,
                "Evaluation": evaluation_text,
                "Fixed_Testcases": final_fixed,
                "LLM_Reasoning": final_reasoning,
                "Method_Inferred": method_name,
                "Leakage_Prevented": True,
                "Buggy_Code_Sent": True,
                "Correct_Code_Sent": False,
                "MI_Buggy": mi_buggy,
                "MI_Correct": mi_correct,
                "Testcases_Original": testcase,
                "Testcases_After_MethodFix": testcase_pre,
                "Code": correct_code,
                "Buggy_Code": buggy_code,
            })

            time.sleep(args.sleep)

    # Write per-row output
    out_df = pd.DataFrame(results)
    out_df.to_csv(args.out_per_row, index=False)

    # Build summary (overall + by dataset + by mode)
    summary_rows: List[Dict[str, Any]] = []

    # Overall by mode
    for mode_name in counters:
        summary_rows.append({
            "Scope": "OVERALL",
            "Dataset": "ALL",
            "Mode": mode_name,
            "Rows_Total": counters[mode_name]["PASS"] + counters[mode_name]["FAIL"] + counters[mode_name]["INVALID"],
            "Rows_Invalid": counters[mode_name]["INVALID"],
            "Rows_Pass": counters[mode_name]["PASS"],
            "Rows_Fail": counters[mode_name]["FAIL"],
            "Leakage_Prevented": True,
            "Model": DEEPSEEK_MODEL,
        })

    # By dataset + mode
    datasets_seen = sorted(set(out_df["Dataset"].tolist()))
    for ds in datasets_seen:
        for mode_name in counters:
            sub = out_df[(out_df["Dataset"] == ds) & (out_df["Mode"] == mode_name)]
            if sub.empty:
                continue
            summary_rows.append({
                "Scope": "BY_DATASET",
                "Dataset": ds,
                "Mode": mode_name,
                "Rows_Total": int(len(sub)),
                "Rows_Invalid": int((sub["Status"].str.startswith("INVALID")).sum()),
                "Rows_Pass": int((sub["Status"] == "PASS").sum()),
                "Rows_Fail": int((sub["Status"] == "FAIL").sum()),
                "Leakage_Prevented": True,
                "Model": DEEPSEEK_MODEL,
            })

    summary = pd.DataFrame(summary_rows)
    summary["Columns_Seen"] = ", ".join([c.strip() for c in original_cols])
    summary.to_csv(args.out_summary, index=False)

    print("DONE")
    print(summary.to_string(index=False))
    print(f"Per-row CSV: {args.out_per_row}")
    print(f"Summary CSV:  {args.out_summary}")


if __name__ == "__main__":
    main()
