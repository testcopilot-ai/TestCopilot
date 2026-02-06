#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import ast
import json
import tempfile
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    from radon.metrics import mi_visit
except Exception:
    mi_visit = None


# =========================================================
# DEFAULT PATHS (EDIT ONCE)
# =========================================================
DEFAULT_INPUT_CSV = "home to directory/leetcode_scenario_generated_testcases.csv"
DEFAULT_OUT_PER_ROW = "home to directory/leetcode_scenario_metrics_per_row.csv"
DEFAULT_OUT_SUMMARY = "home to directoryleetcode_scenario_metrics_summary.csv"


# -----------------------------
# Helpers
# -----------------------------
def safe_str(x: Any) -> str:
    return "" if x is None else str(x)

def is_blank(x: Any) -> bool:
    return safe_str(x).strip() == ""

def write_file(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write((content or "").rstrip() + "\n")

def run_cmd(cmd: List[str], cwd: str, timeout: int = 120) -> Tuple[int, str]:
    p = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    return p.returncode, p.stdout

def shorten(text: str, max_chars: int = 4000) -> str:
    text = text or ""
    return text if len(text) <= max_chars else (text[:max_chars] + "\n...[TRUNCATED]...")

def tool_exists(module: str) -> bool:
    try:
        __import__(module)
        return True
    except Exception:
        return False


# =========================================================
# Parse method signature / method name (FROM YOUR DATASET)
# Your functional_requirement contains either:
#   "Signature:\ndef xxx(...):"
# or "Method Signature:\ndef xxx(...):"
# =========================================================
SIG_NAME_RE = re.compile(r"\bdef\s+([A-Za-z_]\w*)\s*\(")

# accept both "Signature:" and "Method Signature:"
SIG_BLOCK_RE = re.compile(
    r"(?:^|\n)\s*(?:Method\s+Signature|Signature)\s*:\s*(def\s+[A-Za-z_]\w*\s*\(.*?\)\s*:\s*)",
    re.DOTALL,
)

def signature_from_functional_requirement(fr_text: str) -> Optional[str]:
    m = SIG_BLOCK_RE.search(fr_text or "")
    if not m:
        return None
    sig = m.group(1).strip()
    # keep only first line of signature
    sig = sig.splitlines()[0].strip()
    return sig

def method_name_from_signature(sig: str) -> Optional[str]:
    m = SIG_NAME_RE.search(sig or "")
    return m.group(1) if m else None

def infer_target_method_from_solution(solution_code: str) -> Optional[str]:
    """
    Fallback if signature parsing fails: infer likely method by looking for
    class Solution: def <something>(self, ...) and picking the first public method.
    """
    try:
        tree = ast.parse(solution_code)
    except Exception:
        return None

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Solution":
            for child in node.body:
                if isinstance(child, ast.FunctionDef):
                    if child.name.startswith("_"):
                        continue
                    return child.name
    return None


# =========================================================
# AST: function ranges + calls
# =========================================================
@dataclass
class FuncInfo:
    name: str
    start: int
    end: int
    is_method: bool

def _node_end_lineno(node: ast.AST) -> int:
    end = getattr(node, "end_lineno", None)
    if isinstance(end, int):
        return end
    return getattr(node, "lineno", 0) or 0

def extract_functions_and_calls(code: str) -> Tuple[List[FuncInfo], Dict[str, List[str]]]:
    try:
        tree = ast.parse(code)
    except Exception:
        return [], {}

    funcs: List[FuncInfo] = []
    calls: Dict[str, List[str]] = {}

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.stack: List[str] = []

        def visit_FunctionDef(self, node: ast.FunctionDef):
            name = node.name
            start = getattr(node, "lineno", 0) or 0
            end = _node_end_lineno(node)
            funcs.append(FuncInfo(name=name, start=start, end=end, is_method=False))
            self.stack.append(name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
            self.visit_FunctionDef(node)  # treat same

        def visit_ClassDef(self, node: ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    name = child.name
                    start = getattr(child, "lineno", 0) or 0
                    end = _node_end_lineno(child)
                    funcs.append(FuncInfo(name=name, start=start, end=end, is_method=True))
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call):
            if not self.stack:
                return
            caller = self.stack[-1]
            callee = None
            if isinstance(node.func, ast.Name):
                callee = node.func.id
            elif isinstance(node.func, ast.Attribute):
                callee = node.func.attr
            if callee:
                calls.setdefault(caller, []).append(callee)
            self.generic_visit(node)

    Visitor().visit(tree)

    # uniq calls
    for k, v in list(calls.items()):
        uniq: List[str] = []
        for c in v:
            if c not in uniq:
                uniq.append(c)
        calls[k] = uniq

    return funcs, calls


# =========================================================
# Coverage via coverage.py JSON (pytest)
# =========================================================
@dataclass
class CoverageJSON:
    overall_percent: float
    statement_cov: float
    branch_cov: float
    arc_cov: float
    executed_lines: set
    funcs: List[FuncInfo]
    calls: Dict[str, List[str]]
    debug: str

def compute_coverage_json(solution_code: str, test_code: str, timeout: int = 120) -> CoverageJSON:
    funcs, calls = extract_functions_and_calls(solution_code)

    if not tool_exists("coverage") or not tool_exists("pytest"):
        return CoverageJSON(0, 0, 0, 0, set(), funcs, calls, "missing-python-tools: coverage/pytest not installed")

    with tempfile.TemporaryDirectory(prefix="metrics_cov_") as td:
        write_file(os.path.join(td, "solution.py"), solution_code)
        write_file(os.path.join(td, "test_generated.py"), (test_code or "").strip() + "\n")

        rc, out = run_cmd(
            ["python", "-m", "coverage", "run", "--branch", "-m", "pytest", "-q", "test_generated.py"],
            cwd=td,
            timeout=timeout,
        )
        if rc != 0:
            return CoverageJSON(0, 0, 0, 0, set(), funcs, calls, f"coverage-run-failed:\n{shorten(out)}")

        rc2, out2 = run_cmd(["python", "-m", "coverage", "json", "-o", "cov.json"], cwd=td, timeout=timeout)
        if rc2 != 0 or not os.path.exists(os.path.join(td, "cov.json")):
            return CoverageJSON(0, 0, 0, 0, set(), funcs, calls, f"coverage-json-failed:\n{shorten(out2)}")

        with open(os.path.join(td, "cov.json"), "r", encoding="utf-8") as f:
            data = json.load(f)

        files = data.get("files", {})
        sol_key = next((k for k in files.keys() if k.endswith("solution.py")), None)
        if not sol_key:
            return CoverageJSON(0, 0, 0, 0, set(), funcs, calls, "coverage-json-missing-solution.py")

        file_obj = files[sol_key]
        summary = file_obj.get("summary", {}) or {}

        overall_percent = float(summary.get("percent_covered", 0.0) or 0.0)

        num_statements = int(summary.get("num_statements", 0) or 0)
        covered_lines = int(summary.get("covered_lines", 0) or 0)
        statement_cov = (covered_lines / num_statements * 100.0) if num_statements else 0.0

        num_branches = int(summary.get("num_branches", 0) or 0)
        covered_branches = int(summary.get("covered_branches", 0) or 0)
        branch_cov = (covered_branches / num_branches * 100.0) if num_branches else 0.0

        arcs = file_obj.get("arcs", []) or []
        exec_arcs = file_obj.get("executed_arcs", []) or []
        arc_cov = (len(exec_arcs) / len(arcs) * 100.0) if arcs else 0.0

        executed_lines = set(file_obj.get("executed_lines", []) or [])

        return CoverageJSON(
            overall_percent=overall_percent,
            statement_cov=statement_cov,
            branch_cov=branch_cov,
            arc_cov=arc_cov,
            executed_lines=executed_lines,
            funcs=funcs,
            calls=calls,
            debug="",
        )


# =========================================================
# Run tests (FAR / failures) using pytest
# =========================================================
@dataclass
class SuiteRun:
    passed: bool
    tests_run: int
    failing_tests: int
    output: str

def _parse_pytest_counts(out: str) -> Tuple[int, int]:
    """
    Returns: (tests_run, failing_tests)
    Best-effort. If cannot parse and output indicates failure, fallback later.
    """
    out = out or ""
    tests_run = 0
    failing = 0

    m = re.search(r"(\d+)\s+passed", out)
    if m:
        tests_run += int(m.group(1))

    m = re.search(r"(\d+)\s+failed", out)
    if m:
        failing += int(m.group(1))
        tests_run += int(m.group(1))

    # collection/import errors
    m = re.search(r"(\d+)\s+error[s]?", out)
    if m:
        failing += int(m.group(1))
        tests_run += int(m.group(1))

    return tests_run, failing

def run_suite_pytest(solution_code: str, test_code: str, timeout: int = 120) -> SuiteRun:
    if not tool_exists("pytest"):
        return SuiteRun(False, 0, 1, "missing-python-tools: pytest not installed")

    with tempfile.TemporaryDirectory(prefix="metrics_suite_") as td:
        write_file(os.path.join(td, "solution.py"), solution_code)
        write_file(os.path.join(td, "test_generated.py"), (test_code or "").strip() + "\n")

        rc, out = run_cmd(["python", "-m", "pytest", "-q", "test_generated.py"], cwd=td, timeout=timeout)
        passed = (rc == 0)
        tests_run, failing = _parse_pytest_counts(out)

        # IMPORTANT FIX:
        # if pytest failed but we couldn't parse failing count, count at least 1 failing
        if (not passed) and failing == 0:
            failing = 1
            if tests_run == 0:
                tests_run = 1

        return SuiteRun(passed=passed, tests_run=tests_run, failing_tests=failing, output=out)

def extract_failure_reason(output: str) -> str:
    out = output or ""
    if not any(k in out.lower() for k in ["failed", "error", "traceback", "exception", "assert"]):
        return ""
    tail = "\n".join(out.splitlines()[-250:])
    return shorten(tail, 4000)


# =========================================================
# Maintainability Index
# =========================================================
def compute_mi(code: str) -> float:
    if mi_visit is None:
        return 0.0
    try:
        return float(mi_visit(code, multi=True))
    except Exception:
        return 0.0


# =========================================================
# Lightweight Mutation Testing (TCE)
# =========================================================
MUTATIONS = [
    (" == ", " != "),
    (" != ", " == "),
    (" <= ", " < "),
    (" >= ", " > "),
    (" < ", " <= "),
    (" > ", " >= "),
    (" + ", " - "),
    (" - ", " + "),
    (" and ", " or "),
    (" or ", " and "),
    (" True", " False"),
    (" False", " True"),
]

def generate_mutants(code: str, max_mutants: int = 40) -> List[str]:
    mutants: List[str] = []
    for old, new in MUTATIONS:
        start = 0
        while len(mutants) < max_mutants:
            idx = code.find(old, start)
            if idx == -1:
                break
            mutated = code[:idx] + new + code[idx + len(old):]
            if mutated != code and mutated not in mutants:
                try:
                    ast.parse(mutated)
                    mutants.append(mutated)
                except Exception:
                    pass
            start = idx + len(old)
    return mutants

def mutation_score(correct_code: str, test_code: str, timeout: int = 120, max_mutants: int = 40) -> Tuple[int, int]:
    mutants = generate_mutants(correct_code, max_mutants=max_mutants)
    if not mutants:
        return 0, 0
    killed = 0
    for mcode in mutants:
        r = run_suite_pytest(mcode, test_code, timeout=timeout)
        if not r.passed:
            killed += 1
    return len(mutants), killed


# =========================================================
# Extra coverage-derived metrics
# =========================================================
def function_coverage(funcs: List[FuncInfo], executed_lines: set) -> float:
    if not funcs:
        return 0.0
    hit = 0
    for fn in funcs:
        if any((ln in executed_lines) for ln in range(fn.start, fn.end + 1)):
            hit += 1
    return hit / len(funcs) * 100.0

def shallow_deep_coverage(funcs: List[FuncInfo], executed_lines: set, target_method: Optional[str]) -> Tuple[float, float]:
    if not funcs or not target_method:
        return 0.0, 0.0
    target = next((f for f in funcs if f.name == target_method), None)
    if not target:
        return 0.0, 0.0

    def cov_for_range(start: int, end: int) -> float:
        total = max(0, end - start + 1)
        if total == 0:
            return 0.0
        covered = sum(1 for ln in range(start, end + 1) if ln in executed_lines)
        return covered / total * 100.0

    shallow = cov_for_range(target.start, target.end)

    total_other = 0
    covered_other = 0
    for fn in funcs:
        if fn.name == target_method:
            continue
        total_other += max(0, fn.end - fn.start + 1)
        covered_other += sum(1 for ln in range(fn.start, fn.end + 1) if ln in executed_lines)

    deep = (covered_other / total_other * 100.0) if total_other else 0.0
    return shallow, deep

def integration_coverage(funcs: List[FuncInfo], calls: Dict[str, List[str]], executed_lines: set, target_method: Optional[str]) -> float:
    if not target_method or target_method not in calls:
        return 0.0
    fn_map = {f.name: f for f in funcs}
    callees = [c for c in calls.get(target_method, []) if c in fn_map]
    if not callees:
        return 0.0
    reached = 0
    for callee in callees:
        finfo = fn_map[callee]
        if any((ln in executed_lines) for ln in range(finfo.start, finfo.end + 1)):
            reached += 1
    return reached / len(callees) * 100.0


# =========================================================
# Compute metrics per row (MATCH YOUR CSV COLUMN NAMES)
# Columns: Dataset_ID, functional_requirement, Scenarios, Testcases, Code, error, Buggy_Code
# =========================================================
def compute_metrics_for_row(row: pd.Series) -> Dict[str, Any]:
    functional_req = safe_str(row.get("functional_requirement"))
    correct_code = safe_str(row.get("Code"))
    test_code = safe_str(row.get("Testcases"))
    buggy_code = safe_str(row.get("Buggy_Code"))

    method_sig = signature_from_functional_requirement(functional_req) or ""
    target_method = method_name_from_signature(method_sig)

    # Fallback if signature parsing fails
    if not target_method:
        target_method = infer_target_method_from_solution(correct_code)

    debug_notes: List[str] = []
    if not method_sig:
        debug_notes.append("sig-not-found-in-functional_requirement")
    if not target_method:
        debug_notes.append("target-method-not-found")

    # Run suite on correct (false alarms)
    correct_suite = run_suite_pytest(correct_code, test_code, timeout=120)
    passed_correct = 1 if correct_suite.passed else 0
    false_alarms = int(correct_suite.failing_tests)
    fail_reason_correct = extract_failure_reason(correct_suite.output)

    # Run suite on buggy (defect detection)
    known_defects = 1 if not is_blank(buggy_code) else 0
    actual_defects_detected = 0
    buggy_failing_tests = 0
    fail_reason_buggy = ""
    buggy_output = ""

    if known_defects:
        buggy_suite = run_suite_pytest(buggy_code, test_code, timeout=120)
        buggy_output = buggy_suite.output
        buggy_failing_tests = int(buggy_suite.failing_tests)
        fail_reason_buggy = extract_failure_reason(buggy_suite.output)

        # bug detected only if tests pass on correct but fail on buggy
        if correct_suite.passed and (not buggy_suite.passed):
            actual_defects_detected = 1
    else:
        debug_notes.append("no-buggy-code")

    # DDP
    ddp = (actual_defects_detected / known_defects * 100.0) if known_defects else 0.0

    # FAR
    total_failing_tests = false_alarms + buggy_failing_tests
    far = (false_alarms / total_failing_tests * 100.0) if total_failing_tests > 0 else 0.0

    # Coverage
    cov = compute_coverage_json(correct_code, test_code, timeout=120)
    if cov.debug:
        debug_notes.append(cov.debug)

    coverage = round(cov.overall_percent, 2)
    statement_cov = round(cov.statement_cov, 2)
    branch_cov = round(cov.branch_cov, 2)
    path_cov = round(cov.arc_cov, 2)
    coverage_total = round((statement_cov + branch_cov) / 2.0, 2)

    func_cov = round(function_coverage(cov.funcs, cov.executed_lines), 2)
    shallow_cov, deep_cov = shallow_deep_coverage(cov.funcs, cov.executed_lines, target_method)
    shallow_cov = round(shallow_cov, 2)
    deep_cov = round(deep_cov, 2)
    integ_cov = round(integration_coverage(cov.funcs, cov.calls, cov.executed_lines, target_method), 2)

    # TCE
    total_mutants, killed = mutation_score(correct_code, test_code, timeout=120, max_mutants=40)
    tce = round((killed / total_mutants * 100.0) if total_mutants else 0.0, 2)

    # Maintainability
    mi = round(compute_mi(correct_code), 2)

    return {
        "Coverage": coverage,
        "TCE": tce,
        "DDP": round(ddp, 2),
        "FAR": round(far, 2),

        "Function_Coverage": func_cov,
        "Statement_Coverage": statement_cov,
        "Branch_Coverage": branch_cov,
        "Coverage_Total": coverage_total,
        "Path_Coverage": round(path_cov, 2),
        "Shallow_Coverage": shallow_cov,
        "Deep_Coverage": deep_cov,
        "Integration_Coverage": integ_cov,

        "# Bugs Detected": actual_defects_detected,
        "# False Alarms": false_alarms,
        "Known_Defects": known_defects,

        "Passed_Correct": passed_correct,
        "Tests_Run": int(correct_suite.tests_run),
        "Failing_Tests_Correct": int(false_alarms),
        "Failing_Tests_Buggy": int(buggy_failing_tests),

        "Signature_Parsed": method_sig,
        "Target_Method": target_method or "",

        "Fail_Reason_Correct": fail_reason_correct,
        "Fail_Reason_Buggy": fail_reason_buggy,
        "Suite_Output_Correct": shorten(correct_suite.output, 4000),
        "Suite_Output_Buggy": shorten(buggy_output, 4000),

        "Maintainability_Index": mi,
        "Total_Mutants": total_mutants,
        "Debug": " | ".join([x for x in debug_notes if x]),
    }


# =========================================================
# Main
# =========================================================
def main(input_csv: str, out_per_row: str, out_summary: str) -> None:
    df = pd.read_csv(input_csv)

    required = {"Dataset_ID", "functional_requirement", "Testcases", "Code"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    results: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        did = row.get("Dataset_ID")
        print(f"Computing metrics for Dataset_ID={did} ...")
        try:
            res = compute_metrics_for_row(row)
        except Exception as e:
            res = {
                "Coverage": 0.0,
                "TCE": 0.0,
                "DDP": 0.0,
                "FAR": 0.0,
                "# Bugs Detected": 0,
                "# False Alarms": 0,
                "Fail_Reason_Correct": f"metrics-failed: {e}",
                "Debug": f"metrics-failed: {e}",
            }
        results.append(res)

    metrics_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
    metrics_df.to_csv(out_per_row, index=False)
    print(f"\n✅ Saved per-row metrics: {out_per_row}")

    metric_cols = [
        "Coverage", "TCE", "DDP", "FAR",
        "Function_Coverage", "Statement_Coverage", "Branch_Coverage", "Path_Coverage",
        "Shallow_Coverage", "Deep_Coverage", "Integration_Coverage",
        "Maintainability_Index", "# Bugs Detected", "# False Alarms"
    ]

    summary_rows: List[Dict[str, Any]] = [{"item": "Rows_Evaluated", "value": len(metrics_df)}]

    if "# Bugs Detected" in metrics_df.columns:
        summary_rows.append({"item": "TOTAL_Bugs_Detected", "value": int(metrics_df["# Bugs Detected"].sum())})
    if "# False Alarms" in metrics_df.columns:
        summary_rows.append({"item": "TOTAL_False_Alarms", "value": int(metrics_df["# False Alarms"].sum())})

    for c in metric_cols:
        if c in metrics_df.columns:
            summary_rows.append({"item": f"AVG_{c}", "value": float(metrics_df[c].mean())})

    pd.DataFrame(summary_rows).to_csv(out_summary, index=False)
    print(f"✅ Saved summary: {out_summary}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default=DEFAULT_INPUT_CSV)
    parser.add_argument("--out_per_row", default=DEFAULT_OUT_PER_ROW)
    parser.add_argument("--out_summary", default=DEFAULT_OUT_SUMMARY)
    args = parser.parse_args()
    main(args.input_csv, args.out_per_row, args.out_summary)
