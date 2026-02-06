#!/usr/bin/env python3
"""
Metrics runner (per-row + overall summaries) for generated pytest tests against Buggy_Code and Code.

v4 updates:
1) Testcase-level matching via JUnit <testcase> parsing:
   - False_Alarm_Tests: count of testcase fails on Correct
   - Bugs_Detected_Tests: count of testcase fails on Buggy AND passes on Correct
   - FAR = FA / (FA + BD)
   - TCE = 100 * BD / Compared_Tests
   - DDP = 100 * BD / Correct_Pass_Tests  (only considers tests that pass on Correct)

2) Coverage correctness:
   - Function tracing uses sys.setprofile (NOT sys.settrace) so coverage.py line/branch collection works.
   - func_trace.py omitted from coverage.

3) Robust branch coverage parsing across coverage.py JSON schemas.

Usage:
  python3 mainagentmetrics.py --input_csv agent_per_row.csv --out_csv metrics_out.csv
  python3 mainagentmetrics.py --input_csv agent_per_row.csv --out_csv metrics_out.csv --limit 50 --timeout_sec 60
"""

import argparse
import ast
import json
import os
import re
import subprocess
import tempfile
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional

import pandas as pd
from radon.metrics import mi_visit


# =========================
# Helpers
# =========================
def s(x) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)


def strip_code_fences(txt: str) -> str:
    if not txt:
        return ""
    m = re.search(r"```(?:python)?\s*(.*?)\s*```", txt, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return txt.strip()


def fix_pytest_parametrize_double_quotes(testcase: str) -> str:
    # @pytest.mark.parametrize(""a,b"", [ -> @pytest.mark.parametrize("a,b", [
    if not testcase:
        return ""
    return re.sub(
        r'@pytest\.mark\.parametrize\(\s*""(.*?)""\s*,',
        r'@pytest.mark.parametrize("\1",',
        testcase,
        flags=re.DOTALL,
    )


def ensure_single_solution_import(testcase: str) -> str:
    """Keep exactly one: from solution import Solution"""
    if not testcase:
        return ""
    lines = testcase.splitlines()
    seen = False
    out = []
    for ln in lines:
        if ln.strip() == "from solution import Solution":
            if seen:
                continue
            seen = True
        out.append(ln)
    if not seen:
        out.insert(0, "from solution import Solution")
    return "\n".join(out).strip() + "\n"


def is_python_syntax_ok(src: str) -> Tuple[bool, str]:
    if not src or not src.strip():
        return False, "empty"
    try:
        ast.parse(src)
        return True, ""
    except Exception as e:
        return False, str(e)


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


# =========================
# Function coverage tracing (separate file, omitted from coverage)
# IMPORTANT: sys.setprofile so we do NOT override coverage.py sys.settrace
# =========================
FUNC_TRACE_MODULE = r"""
import sys, json, atexit
EXEC_FUNCS = set()

def _prof(frame, event, arg):
    if event == "call":
        code = frame.f_code
        fn = code.co_name
        mod = frame.f_globals.get("__name__", "")
        if mod == "solution":
            if "self" in frame.f_locals and frame.f_locals["self"].__class__.__name__ == "Solution":
                EXEC_FUNCS.add(fn)
    return _prof

sys.setprofile(_prof)

def _dump():
    try:
        with open("func_trace.json", "w", encoding="utf-8") as f:
            json.dump(sorted(list(EXEC_FUNCS)), f)
    except Exception:
        pass

atexit.register(_dump)
"""


def extract_solution_functions(solution_code: str) -> List[str]:
    """Extract method names inside class Solution."""
    try:
        tree = ast.parse(solution_code)
    except Exception:
        return []
    funcs: List[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Solution":
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    funcs.append(item.name)
    return funcs


# =========================
# JUnit parsing
# =========================
def parse_junit_counts(junit_path: str) -> Tuple[int, int, int, int]:
    """Returns (tests, failures, errors, skipped)."""
    if not os.path.exists(junit_path):
        return 0, 0, 0, 0

    try:
        tree = ET.parse(junit_path)
        root = tree.getroot()

        def suite_counts(ts) -> Tuple[int, int, int, int]:
            t = int(ts.attrib.get("tests", 0))
            f = int(ts.attrib.get("failures", 0))
            e = int(ts.attrib.get("errors", 0))
            sk = int(ts.attrib.get("skipped", ts.attrib.get("disabled", 0) or 0))
            return t, f, e, sk

        if root.tag == "testsuite":
            return suite_counts(root)

        if root.tag == "testsuites":
            tests = failures = errors = skipped = 0
            for ts in root.findall("testsuite"):
                t, f, e, sk = suite_counts(ts)
                tests += t
                failures += f
                errors += e
                skipped += sk
            return tests, failures, errors, skipped

    except Exception:
        return 0, 0, 0, 0

    return 0, 0, 0, 0


def parse_junit_testcases(junit_path: str) -> Dict[str, str]:
    """
    Returns mapping: testcase_id -> status
      status in {"pass","fail","error","skipped"}

    testcase_id uses: "{classname}::{name}" (classname may be empty).
    """
    results: Dict[str, str] = {}
    if not os.path.exists(junit_path):
        return results

    try:
        tree = ET.parse(junit_path)
        root = tree.getroot()

        # collect all <testcase> nodes under any testsuite(s)
        testcases = root.findall(".//testcase")
        for tc in testcases:
            name = tc.attrib.get("name", "").strip()
            classname = tc.attrib.get("classname", "").strip()
            tc_id = f"{classname}::{name}" if classname else name

            # Determine status by children tags
            status = "pass"
            if tc.find("skipped") is not None:
                status = "skipped"
            elif tc.find("failure") is not None:
                status = "fail"
            elif tc.find("error") is not None:
                status = "error"

            # If duplicate ids appear, keep the "worst" status deterministically
            # order: error > fail > skipped > pass
            rank = {"pass": 0, "skipped": 1, "fail": 2, "error": 3}
            if tc_id in results:
                if rank[status] > rank[results[tc_id]]:
                    results[tc_id] = status
            else:
                results[tc_id] = status

    except Exception:
        return {}

    return results


# =========================
# Data model
# =========================
@dataclass
class RunResult:
    ok: bool
    exit_code: int
    stderr: str
    stdout: str
    statement_cov: float
    branch_cov: float
    path_cov: float
    function_cov: float
    status: str  # OK / FAIL / TIMEOUT / SYNTAX_ERROR

    # test counts (from suite attrs)
    tests: int
    failures: int
    errors: int
    skipped: int

    # testcase-level statuses
    testcase_status: Dict[str, str]

    @property
    def failed_total(self) -> int:
        return int(self.failures) + int(self.errors)


# =========================
# File builders
# =========================
def make_solution_file(code: str) -> str:
    code = strip_code_fences(code).strip()
    return "import func_trace\n\n" + code + "\n"


def make_test_file(test_code: str) -> str:
    test_code = strip_code_fences(test_code)
    test_code = fix_pytest_parametrize_double_quotes(test_code)
    test_code = ensure_single_solution_import(test_code)
    return test_code.strip() + "\n"


# =========================
# Coverage JSON parsing helpers
# =========================
def _len_if_list(x) -> int:
    return len(x) if isinstance(x, list) else 0


def compute_branch_coverage_from_covjson(sol_file_obj: Dict[str, Any]) -> float:
    """
    Robust branch coverage across coverage.py JSON schemas.

    Order:
    1) summary.num_branches + summary.covered_branches
    2) executed_branches + missing_branches
    3) executed_arcs + missing_arcs
    """
    summ = sol_file_obj.get("summary", {}) if isinstance(sol_file_obj, dict) else {}

    try:
        total_br = int(summ.get("num_branches", 0))
        cov_br = int(summ.get("covered_branches", 0))
        if total_br > 0:
            return (cov_br / total_br) * 100.0
    except Exception:
        pass

    executed_br = sol_file_obj.get("executed_branches")
    missing_br = sol_file_obj.get("missing_branches")
    total = _len_if_list(executed_br) + _len_if_list(missing_br)
    if total > 0:
        return (_len_if_list(executed_br) / total) * 100.0

    executed_arcs = sol_file_obj.get("executed_arcs")
    missing_arcs = sol_file_obj.get("missing_arcs")
    total = _len_if_list(executed_arcs) + _len_if_list(missing_arcs)
    if total > 0:
        return (_len_if_list(executed_arcs) / total) * 100.0

    return 0.0


# =========================
# Runner
# =========================
def run_cmd(cmd: List[str], cwd: str, timeout_sec: int) -> Tuple[int, str, str, bool]:
    env = os.environ.copy()
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

    try:
        p = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=env,
        )
        return p.returncode, p.stdout, p.stderr, False
    except subprocess.TimeoutExpired as e:
        out = e.stdout or ""
        err = e.stderr or ""
        return 124, out, (err + "\nTIMEOUT"), True


def run_pytest_cov(workdir: str, timeout_sec: int) -> RunResult:
    """
    Runs:
      python -m coverage run --branch --source=. --omit=func_trace.py -m pytest -q test_generated.py --junitxml=junit.xml
      python -m coverage json -o cov.json
    """
    py = sys.executable
    junit_path = os.path.join(workdir, "junit.xml")

    cmd_run = [
        py, "-m", "coverage", "run",
        "--branch",
        "--source=.",
        "--omit", "func_trace.py",
        "-m", "pytest", "-q", "test_generated.py",
        "--junitxml", "junit.xml",
    ]
    rc, out, err, timed_out = run_cmd(cmd_run, cwd=workdir, timeout_sec=timeout_sec)

    if timed_out:
        return RunResult(
            ok=False,
            exit_code=rc,
            stderr=err,
            stdout=out,
            statement_cov=0.0,
            branch_cov=0.0,
            path_cov=0.0,
            function_cov=0.0,
            status="TIMEOUT",
            tests=0, failures=0, errors=0, skipped=0,
            testcase_status={},
        )

    ok = (rc == 0)

    # Parse junit
    tests, failures, errors, skipped = parse_junit_counts(junit_path)
    tc_map = parse_junit_testcases(junit_path)

    # coverage json report (best-effort)
    cmd_json = [py, "-m", "coverage", "json", "-o", "cov.json"]
    run_cmd(cmd_json, cwd=workdir, timeout_sec=min(15, timeout_sec))

    statement_cov = 0.0
    branch_cov = 0.0
    path_cov = 0.0

    try:
        with open(os.path.join(workdir, "cov.json"), "r", encoding="utf-8") as f:
            cj = json.load(f)

        sol_obj = cj.get("files", {}).get("solution.py", {})
        summ = sol_obj.get("summary", {})

        statement_cov = float(summ.get("percent_covered", 0.0))
        branch_cov = compute_branch_coverage_from_covjson(sol_obj)
        path_cov = (statement_cov + branch_cov) / 2.0

    except Exception:
        statement_cov = 0.0
        branch_cov = 0.0
        path_cov = 0.0

    # Function coverage from func_trace.json (best-effort)
    function_cov = 0.0
    try:
        with open(os.path.join(workdir, "solution.py"), "r", encoding="utf-8") as f:
            sol_code = f.read()

        funcs = extract_solution_functions(sol_code)
        total_funcs = len(funcs)

        with open(os.path.join(workdir, "func_trace.json"), "r", encoding="utf-8") as f:
            executed = set(json.load(f))

        covered_funcs = len([fn for fn in funcs if fn in executed])
        function_cov = (covered_funcs / total_funcs * 100.0) if total_funcs else 0.0
    except Exception:
        function_cov = 0.0

    return RunResult(
        ok=ok,
        exit_code=rc,
        stderr=err,
        stdout=out,
        statement_cov=round(statement_cov, 2),
        branch_cov=round(branch_cov, 2),
        path_cov=round(path_cov, 2),
        function_cov=round(function_cov, 2),
        status="OK" if ok else "FAIL",
        tests=tests, failures=failures, errors=errors, skipped=skipped,
        testcase_status=tc_map,
    )


def eval_variant(solution_code: str, test_code: str, timeout_sec: int) -> Tuple[RunResult, float, bool, str]:
    mi = -1.0
    try:
        mi = round(float(mi_visit(strip_code_fences(solution_code), multi=True)), 2)
    except Exception:
        mi = -1.0

    with tempfile.TemporaryDirectory(prefix="metric_run_") as td:
        sol_path = os.path.join(td, "solution.py")
        test_path = os.path.join(td, "test_generated.py")
        trace_path = os.path.join(td, "func_trace.py")

        with open(trace_path, "w", encoding="utf-8") as f:
            f.write(FUNC_TRACE_MODULE.strip() + "\n")

        with open(sol_path, "w", encoding="utf-8") as f:
            f.write(make_solution_file(solution_code))

        with open(test_path, "w", encoding="utf-8") as f:
            f.write(make_test_file(test_code))

        sol_src = open(sol_path, encoding="utf-8").read()
        test_src = open(test_path, encoding="utf-8").read()

        ok_sol, err_sol = is_python_syntax_ok(sol_src)
        ok_test, err_test = is_python_syntax_ok(test_src)

        if not ok_sol or not ok_test:
            rr = RunResult(
                ok=False,
                exit_code=2,
                stderr=f"SYNTAX_ERROR: solution_ok={ok_sol}({err_sol}); test_ok={ok_test}({err_test})",
                stdout="",
                statement_cov=0.0,
                branch_cov=0.0,
                path_cov=0.0,
                function_cov=0.0,
                status="SYNTAX_ERROR",
                tests=0, failures=0, errors=0, skipped=0,
                testcase_status={},
            )
            return rr, mi, ok_test, err_test

        rr = run_pytest_cov(td, timeout_sec=timeout_sec)
        return rr, mi, True, ""


# =========================
# Metric calculations
# =========================
def overall_cov(func_cov: float, stmt_cov: float, br_cov: float, path_cov: float) -> float:
    return round((func_cov + stmt_cov + br_cov + path_cov) / 4.0, 2)


def shallow_cov(stmt_cov: float, func_cov: float) -> float:
    return round((stmt_cov + func_cov) / 2.0, 2)


def deep_cov(br_cov: float, path_cov: float) -> float:
    return round((br_cov + path_cov) / 2.0, 2)


def valid_for_semantic_metrics(rr: RunResult) -> bool:
    if rr.status in ("SYNTAX_ERROR", "TIMEOUT"):
        return False
    if rr.tests <= 0:
        return False
    return True


def integration_pass_rate(correct_rr: RunResult) -> float:
    if not valid_for_semantic_metrics(correct_rr):
        return float("nan")
    failed = correct_rr.failed_total
    tests = max(int(correct_rr.tests), 1)
    return round(100.0 * (1.0 - (failed / tests)), 2)


def testcase_level_counts(buggy_rr: RunResult, corr_rr: RunResult) -> Tuple[int, int, int, int]:
    """
    Returns:
      (compared_tests, correct_pass_tests, bugs_detected_tests, false_alarm_tests)

    Definitions (per testcase id match):
      false_alarm_tests: testcase fails/errors on Correct
      bugs_detected_tests: testcase fails/errors on Buggy AND passes on Correct
      correct_pass_tests: testcase passes on Correct
      compared_tests: number of testcase ids compared (intersection)
    """
    bmap = buggy_rr.testcase_status or {}
    cmap = corr_rr.testcase_status or {}

    keys = sorted(set(bmap.keys()) & set(cmap.keys()))
    compared = len(keys)

    correct_pass = 0
    bugs_detected = 0
    false_alarm = 0

    for k in keys:
        cs = cmap.get(k)
        bs = bmap.get(k)

        # treat "fail" and "error" as failure signals
        c_fail = cs in ("fail", "error")
        b_fail = bs in ("fail", "error")

        if cs == "pass":
            correct_pass += 1
            if b_fail:
                bugs_detected += 1
        elif c_fail:
            false_alarm += 1
        else:
            # skipped or unknown: doesn't contribute to pass/fail metrics
            pass

    return compared, correct_pass, bugs_detected, false_alarm


def far_from_counts(bugs_detected_tests: int, false_alarm_tests: int) -> float:
    denom = bugs_detected_tests + false_alarm_tests
    return round((false_alarm_tests / denom) if denom else 0.0, 4)


def tce_from_counts(bugs_detected_tests: int, compared_tests: int) -> float:
    return round(100.0 * (bugs_detected_tests / compared_tests), 4) if compared_tests else 0.0


def ddp_from_counts(bugs_detected_tests: int, correct_pass_tests: int) -> float:
    # detection rate among "clean" tests (those that pass on Correct)
    return round(100.0 * (bugs_detected_tests / correct_pass_tests), 4) if correct_pass_tests else 0.0


# =========================
# Optional: error-type classification (helps debugging FAR quickly)
# =========================
def classify_stderr(stderr: str) -> str:
    if not stderr:
        return ""
    patterns = [
        ("AssertionError", r"AssertionError"),
        ("TypeError", r"TypeError"),
        ("AttributeError", r"AttributeError"),
        ("IndexError", r"IndexError"),
        ("ValueError", r"ValueError"),
        ("ImportError", r"ImportError|ModuleNotFoundError"),
        ("SyntaxError", r"SyntaxError"),
        ("RecursionError", r"RecursionError"),
        ("ZeroDivisionError", r"ZeroDivisionError"),
        ("KeyError", r"KeyError"),
        ("Timeout", r"TIMEOUT"),
    ]
    for name, pat in patterns:
        if re.search(pat, stderr):
            return name
    return "Other"


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--out_csv", default="metrics_out.csv")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--timeout_sec", type=int, default=60, help="Timeout per pytest run (buggy or correct).")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    if args.limit:
        df = df.head(args.limit)

    out_rows: List[Dict[str, Any]] = []
    total = len(df)
    t0 = time.time()

    for i, (_, r) in enumerate(df.iterrows(), start=1):
        dataset_id = s(r.get("Dataset_ID"))
        print(f"[{i}/{total}] Dataset_ID={dataset_id} ...", flush=True)

        tests = s(r.get("Fixed_Testcases"))
        buggy = s(r.get("Buggy_Code"))
        correct = s(r.get("Code"))

        buggy_rr, _buggy_mi, ok_test_buggy, err_test_buggy = eval_variant(
            buggy, tests, timeout_sec=args.timeout_sec
        )
        corr_rr, corr_mi, ok_test_corr, err_test_corr = eval_variant(
            correct, tests, timeout_sec=args.timeout_sec
        )

        tests_syntax_ok = bool(ok_test_buggy and ok_test_corr)
        tests_syntax_err = ""
        if not tests_syntax_ok:
            tests_syntax_err = err_test_corr or err_test_buggy or "unknown test syntax error"

        # Coverage metrics: use Correct run coverage only if syntax OK
        if tests_syntax_ok:
            func_cov = corr_rr.function_cov
            stmt_cov = corr_rr.statement_cov
            br_cov = corr_rr.branch_cov
            path_cov = corr_rr.path_cov

            cov = overall_cov(func_cov, stmt_cov, br_cov, path_cov)
            shallow = shallow_cov(stmt_cov, func_cov)
            deep = deep_cov(br_cov, path_cov)
            mi_val = corr_mi
        else:
            func_cov = float("nan")
            stmt_cov = float("nan")
            br_cov = float("nan")
            path_cov = float("nan")
            cov = float("nan")
            shallow = float("nan")
            deep = float("nan")
            mi_val = float("nan")

        # Semantic metrics validity
        semantic_valid = valid_for_semantic_metrics(buggy_rr) and valid_for_semantic_metrics(corr_rr)

        if semantic_valid:
            compared_tests, correct_pass_tests, bugs_detected_tests, false_alarm_tests = testcase_level_counts(
                buggy_rr, corr_rr
            )
            far = far_from_counts(bugs_detected_tests, false_alarm_tests)
            tce = tce_from_counts(bugs_detected_tests, compared_tests)
            ddp = ddp_from_counts(bugs_detected_tests, correct_pass_tests)
            integ = integration_pass_rate(corr_rr)
        else:
            compared_tests = float("nan")
            correct_pass_tests = float("nan")
            bugs_detected_tests = float("nan")
            false_alarm_tests = float("nan")
            far = float("nan")
            tce = float("nan")
            ddp = float("nan")
            integ = float("nan")

        out_row: Dict[str, Any] = {
            "Dataset_ID": dataset_id,

            "Tests_Syntax_OK": tests_syntax_ok,
            "Tests_Syntax_Error": tests_syntax_err,

            "Semantic_Valid": semantic_valid,  # valid for FAR/TCE/DDP/Integration

            # Main metrics
            "Coverage": cov,
            "TCE": tce,
            "DDP": ddp,
            "FAR": far,

            "Function_Coverage": func_cov,
            "Statement_Coverage": stmt_cov,
            "Branch_Coverage": br_cov,
            "Path_Coverage": path_cov,

            "Shallow_Coverage": shallow,
            "Deep_Coverage": deep,
            "Integration_Coverage": integ,

            "Maintainability_Index": mi_val,

            # Raw junit suite counts
            "Buggy_Pass": buggy_rr.ok,
            "Correct_Pass": corr_rr.ok,

            "Buggy_Tests": buggy_rr.tests,
            "Buggy_Failures": buggy_rr.failures,
            "Buggy_Errors": buggy_rr.errors,
            "Buggy_Skipped": buggy_rr.skipped,
            "Buggy_Failed_Total": buggy_rr.failed_total,

            "Correct_Tests": corr_rr.tests,
            "Correct_Failures": corr_rr.failures,
            "Correct_Errors": corr_rr.errors,
            "Correct_Skipped": corr_rr.skipped,
            "Correct_Failed_Total": corr_rr.failed_total,

            # testcase-level counts
            "Compared_Tests": compared_tests,
            "Correct_Pass_Tests": correct_pass_tests,
            "Bugs_Detected_Tests": bugs_detected_tests,
            "False_Alarm_Tests": false_alarm_tests,

            "Buggy_Status": buggy_rr.status,
            "Correct_Status": corr_rr.status,
            "Buggy_ExitCode": buggy_rr.exit_code,
            "Correct_ExitCode": corr_rr.exit_code,

            "Buggy_Error_Type": classify_stderr(buggy_rr.stderr),
            "Correct_Error_Type": classify_stderr(corr_rr.stderr),

            "Buggy_Stderr": (buggy_rr.stderr or "")[:2000],
            "Correct_Stderr": (corr_rr.stderr or "")[:2000],
        }

        # Keep originals
        out_row["Buggy_Code"] = buggy
        out_row["Code"] = correct
        out_row["Fixed_Testcases"] = tests

        out_rows.append(out_row)

        if i % 10 == 0:
            pd.DataFrame(out_rows).to_csv(args.out_csv, index=False)
            print(f"  saved checkpoint -> {args.out_csv}", flush=True)

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(args.out_csv, index=False)

    # =========================
    # OVERALL SUMMARY
    # =========================
    total_rows = len(out_df)
    syntax_df = out_df[out_df["Tests_Syntax_OK"] == True].copy()
    syntax_rows = len(syntax_df)

    semantic_df = out_df[out_df["Semantic_Valid"] == True].copy()
    semantic_rows = len(semantic_df)

    cov_cols = [
        "Coverage",
        "Function_Coverage", "Statement_Coverage", "Branch_Coverage", "Path_Coverage",
        "Shallow_Coverage", "Deep_Coverage",
        "Maintainability_Index",
    ]
    semantic_cols = ["TCE", "DDP", "FAR", "Integration_Coverage"]
    sum_cols = ["Bugs_Detected_Tests", "False_Alarm_Tests", "Compared_Tests", "Correct_Pass_Tests"]

    for c in cov_cols + semantic_cols + sum_cols:
        if c in syntax_df.columns:
            syntax_df[c] = to_num(syntax_df[c])
        if c in semantic_df.columns:
            semantic_df[c] = to_num(semantic_df[c])

    total_bugs = float(semantic_df["Bugs_Detected_Tests"].sum(skipna=True)) if semantic_rows else 0.0
    total_false_alarms = float(semantic_df["False_Alarm_Tests"].sum(skipna=True)) if semantic_rows else 0.0

    avg_cov_series = syntax_df[cov_cols].mean(numeric_only=True) if syntax_rows else pd.Series(dtype=float)
    avg_sem_series = semantic_df[semantic_cols].mean(numeric_only=True) if semantic_rows else pd.Series(dtype=float)

    print("\n===== OVERALL SUMMARY =====")
    print(f"Total rows: {total_rows}")
    print(f"Valid rows (Tests_Syntax_OK=True): {syntax_rows}")
    print(f"Rows valid for semantic metrics (Semantic_Valid=True): {semantic_rows}")
    print(f"TOTAL Bugs Detected (testcase-level, semantic rows): {total_bugs}")
    print(f"TOTAL False Alarms (testcase-level, semantic rows): {total_false_alarms}")

    if syntax_rows:
        print("\n----- AVERAGES (Tests_Syntax_OK rows) -----")
        for k, v in avg_cov_series.items():
            print(f"{k}: {v:.6f}")

    if semantic_rows:
        print("\n----- AVERAGES (Semantic_Valid rows only) -----")
        for k, v in avg_sem_series.items():
            print(f"{k}: {v:.6f}")
    else:
        print("\nNo Semantic_Valid rows: many TIMEOUT/SYNTAX or pytest collected 0 tests.")

    summary_path = os.path.splitext(args.out_csv)[0] + "_summary.csv"
    summary_items = [
        {"Metric": "Total_Rows", "Average": float(total_rows)},
        {"Metric": "Valid_Rows_Tests_Syntax_OK", "Average": float(syntax_rows)},
        {"Metric": "Valid_Rows_Semantic_Valid", "Average": float(semantic_rows)},
        {"Metric": "TOTAL_Bugs_Detected_Tests", "Average": float(total_bugs)},
        {"Metric": "TOTAL_False_Alarm_Tests", "Average": float(total_false_alarms)},
    ]

    if syntax_rows:
        for k, v in avg_cov_series.items():
            summary_items.append({"Metric": k, "Average": float(v)})

    if semantic_rows:
        for k, v in avg_sem_series.items():
            summary_items.append({"Metric": k, "Average": float(v)})

    pd.DataFrame(summary_items).to_csv(summary_path, index=False)
    print(f"\nSaved summary -> {summary_path}")

    dt = round(time.time() - t0, 2)
    print(f"\nDONE -> {args.out_csv}  (rows={total_rows}, seconds={dt})")


if __name__ == "__main__":
    main()
