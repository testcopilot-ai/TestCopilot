#!/usr/bin/env python3
from __future__ import annotations

import ast
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

# =========================================================
# CONFIG
# =========================================================

DEFAULT_INPUT_CSV = "Home to directory/Zeroshot_leetcode_detailed_results.csv"
DEFAULT_OUTPUT_CSV = "Home to directiory/leetcode_scenario_generated_testcases.csv"


# ========================================================= 
#  CONFIG (HARDCODED – copy/paste) 
# ========================================================= 
DEEPSEEK_API_KEY = "Your_aPI_KEY" 
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1" 
DEEPSEEK_MODEL = "deepseek-coder"

# =========================================================
# DeepSeek Client
# =========================================================

class DeepSeekClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: int = 120,
        max_retries: int = 3,
        retry_sleep: float = 2.0,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "messages": messages, "temperature": temperature}

        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                if r.status_code != 200:
                    raise RuntimeError(f"DeepSeek API error {r.status_code}: {r.text[:1200]}")
                return r.json()["choices"][0]["message"]["content"]
            except Exception as e:
                last_err = e
                time.sleep(self.retry_sleep * attempt)
        raise RuntimeError(f"DeepSeek request failed after retries: {last_err}")

# =========================================================
# Helpers
# =========================================================

SIG_NAME_RE = re.compile(r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
TEST_DEF_RE = re.compile(r"^\s*def\s+(test_[A-Za-z0-9_]+)\s*\(.*?\)\s*:\s*$", re.M)

def method_name_from_signature(sig: str) -> str:
    m = SIG_NAME_RE.search(sig or "")
    return m.group(1) if m else "generated"

def extract_one_example_test(ranked_test_case: Any) -> str:
    if ranked_test_case is None:
        return ""
    s = str(ranked_test_case).strip()
    if not s:
        return ""

    m = TEST_DEF_RE.search(s)
    if not m:
        return "\n".join(s.splitlines()[:60]).strip()

    start = m.start()
    m2 = TEST_DEF_RE.search(s[m.end():])
    end = len(s) if not m2 else (m.end() + m2.start())
    snippet = s[start:end].strip()
    return "\n".join(snippet.splitlines()[:120]).strip()

def list_additional_functions(code: str, target_name: str) -> List[str]:
    """
    Return helper function names in the code excluding target method and dunder.
    For LeetCode style, helpers are often methods in class Solution.
    """
    try:
        tree = ast.parse(code)
    except Exception:
        return []

    helpers: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            name = node.name
            if name == target_name:
                continue
            if name.startswith("__") and name.endswith("__"):
                continue
            helpers.append(name)
    # keep unique, stable order
    uniq = []
    for h in helpers:
        if h not in uniq:
            uniq.append(h)
    return uniq

def build_functional_requirement_block(fr: str, sig: str, one_test: str, additional: List[str]) -> str:
    fr = (fr or "").strip()
    sig = (sig or "").strip()
    one_test = (one_test or "").strip() or "[]"
    additional_str = f"[{', '.join(additional)}]" if additional else "[]"

    instructions = (
        "Generate 3-5 unique test case scenarios based on the above details.\n"
        "Each scenario should include:\n"
        "- A brief description of the test purpose.\n"
        "- Variations in input parameters or edge cases.\n"
        "- Expected outcome."
    )

    return (
        "Functional Requirement:\n"
        f"{fr}\n\n"
        "Signature:\n"
        f"{sig}\n\n"
        "Test Case:\n"
        f"{one_test}\n\n"
        "Additional Functions:\n"
        f"{additional_str}\n\n"
        "Instructions:\n"
        f"{instructions}\n"
    ).strip()

def build_generation_prompt(fr_block: str, method_name: str, signature: str) -> List[Dict[str, str]]:
    system = (
        "You are an expert Python test designer.\n"
        "Return plain text ONLY.\n"
        "Do NOT output JSON.\n"
        "Do NOT use markdown fences.\n"
        "Output EXACTLY two sections with these exact headers (each on its own line):\n"
        "Scenarios:\n"
        "Testcases:\n"
        "No other headers."
    )

    user = f"""
Use the following spec to generate NEW scenarios and testcases.

{fr_block}

Rules:
Scenarios:
- Write 10 to 14 scenarios, numbered "Test Case 1", "Test Case 2", ...
- Each must include: Purpose, Input, Expected Output.
- Keep inputs valid for the signature.

Testcases:
- Output PYTEST code (not unittest).
- Must use pytest parametrization so that each scenario runs independently.
- Structure MUST be exactly:
    import pytest
    from solution import Solution

    @pytest.mark.parametrize("...", [...])
    def test_{method_name}(...):
        sol = Solution()
        assert sol.{method_name}(...) == expected

- Do NOT print.
- Do NOT use randomness.
- Use only stdlib + pytest.
- IMPORTANT: Call ONLY sol.{method_name} (do not call a free function).
- Match parameter order/count to this signature:
{signature}
""".strip()

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def split_llm_sections(text: str) -> Tuple[str, str]:
    t = (text or "").strip().replace("\r\n", "\n")
    m1 = re.search(r"(?im)^\s*Scenarios\s*:\s*$", t)
    m2 = re.search(r"(?im)^\s*Testcases\s*:\s*$", t)
    if not m1 or not m2 or m2.start() < m1.start():
        raise RuntimeError("Model output missing required 'Scenarios:' / 'Testcases:' headers.")
    scenarios = t[m1.end():m2.start()].strip()
    testcases = t[m2.end():].strip()
    return scenarios, testcases

def is_valid_python(code: str) -> Tuple[bool, Optional[str]]:
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"{e.msg} (line {e.lineno}, col {e.offset})"

def strip_print_lines(code: str) -> str:
    lines = []
    for ln in (code or "").splitlines():
        if re.search(r"^\s*print\s*\(", ln):
            continue
        lines.append(ln)
    return "\n".join(lines).strip()

def force_test_function_name(code: str, wanted_name: str) -> str:
    # Replace first "def test_x(" with "def test_<wanted_name>("
    return re.sub(r"^\s*def\s+test_[A-Za-z0-9_]+\s*\(",
                  f"def test_{wanted_name}(",
                  code, count=1, flags=re.M)

def force_calls_use_solution(code: str, method_name: str) -> str:
    """
    Ensure calls are sol.method_name(...), not bare method_name(...).
    Avoid touching occurrences like "Solution.method_name" or "sol.method_name".
    """
    pat = re.compile(rf"(?<!\.)\b{re.escape(method_name)}\s*\(")
    return pat.sub(f"sol.{method_name}(", code)

def wrap_as_module(test_func: str) -> str:
    return (
        "from solution import Solution\n\n"
        f"{test_func.strip()}\n"
    ).strip() + "\n"

# =========================================================
# Per-row generation
# =========================================================

def generate_for_row(
    client: DeepSeekClient,
    dataset_id: Any,
    fr: str,
    sig: str,
    code_to_copy: str,
    ranked_test_case: Any,
) -> Dict[str, Any]:
    method_name = method_name_from_signature(sig)
    one_test = extract_one_example_test(ranked_test_case)
    additional = list_additional_functions(code_to_copy or "", method_name)

    fr_block = build_functional_requirement_block(fr, sig, one_test, additional)

    raw = client.chat(build_generation_prompt(fr_block, method_name, sig), temperature=0.2)
    scenarios, testcases = split_llm_sections(raw)

    # sanitize + enforce
    testcases = strip_print_lines(testcases)
    testcases = force_test_function_name(testcases, method_name)
    testcases = force_calls_use_solution(testcases, method_name)

    module_code = wrap_as_module(testcases)
    ok, err = is_valid_python(module_code)
    if not ok:
        raise RuntimeError(f"Invalid python in generated Testcases: {err}")

    return {
        "Dataset_ID": dataset_id,
        "functional_requirement": fr_block,
        "Scenarios": scenarios,
        "Testcases": module_code.strip(),
        "Code": (code_to_copy or "").strip(),
        "error": "",
    }

# =========================================================
# CSV pipeline
# =========================================================

def generate_dataset_from_csv(input_csv: str, output_csv: str, start: int = 0, limit: Optional[int] = None) -> None:
    df = pd.read_csv(input_csv)

    required = {"Dataset_ID", "Functional_Requirement", "Method_Signature", "Ranked_Test_Case"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    # copy correct code
    if "Correct_Code" in df.columns:
        code_col = "Correct_Code"
    elif "Code" in df.columns:
        code_col = "Code"
    else:
        raise ValueError("Input CSV must include either 'Correct_Code' or 'Code' column.")

    if not DEEPSEEK_API_KEY:
        raise SystemExit("ERROR: Set DEEPSEEK_API_KEY as an environment variable (do NOT hardcode it).")

    client = DeepSeekClient(DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL)

    end = len(df) if limit is None else min(len(df), start + limit)
    out_rows: List[Dict[str, Any]] = []

    for i in range(start, end):
        row = df.iloc[i]
        did = row["Dataset_ID"]
        print(f"[{i+1}/{end}] Processing Dataset_ID={did}")

        try:
            out_row = generate_for_row(
                client=client,
                dataset_id=did,
                fr=str(row["Functional_Requirement"]),
                sig=str(row["Method_Signature"]),
                code_to_copy=str(row[code_col]),
                ranked_test_case=row.get("Ranked_Test_Case", ""),
            )
        except Exception as e:
            out_row = {
                "Dataset_ID": did,
                "functional_requirement": build_functional_requirement_block(
                    str(row["Functional_Requirement"]),
                    str(row["Method_Signature"]),
                    extract_one_example_test(row.get("Ranked_Test_Case", "")),
                    list_additional_functions(str(row[code_col]), method_name_from_signature(str(row["Method_Signature"]))),
                ),
                "Scenarios": "",
                "Testcases": "",
                "Code": str(row[code_col]).strip(),
                "error": str(e),
            }
            print(f"  ⚠️ Failed Dataset_ID={did}: {e}")

        out_rows.append(out_row)

        if (i - start + 1) % 10 == 0:
            pd.DataFrame(out_rows).to_csv(output_csv, index=False)
            print(f"  💾 checkpoint saved to {output_csv}")

    pd.DataFrame(out_rows).to_csv(output_csv, index=False)
    print(f"\n✅ Saved output CSV: {output_csv}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output_csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    generate_dataset_from_csv(args.input_csv, args.output_csv, args.start, args.limit)
