#!/usr/bin/env python
# -*- coding: utf-8 -*-
# """
# compute_all_metrics.py — fast metrics for your generated tests (Windows-safe, chunked)
#
# Features:
# - Avoids WinError 206 by running pytest in CHUNKS of nodeids.
# - Combines coverage across chunks into one metrics_out/coverage.json.
# - Optional overall wall-time cap, and optional per-CHUNK timeout (works even without plugins).
# - Optional per-TEST timeout (needs pytest-timeout plugin).
# - Optional parallel execution (needs pytest-xdist).
# - Optional sampling to keep runs quick.
# - Computes function/statement/branch/path/shallow/deep coverage, averages per file, MI, TCE/DDP (if mutation JSON provided).
#
# Requires:
#   pip install pytest pytest-cov coverage
# Optional:
#   pip install pytest-xdist pytest-timeout radon
#
# Typical fast run:
#   python compute_all_metrics.py --parallel auto --per-test-timeout 2 --chunk-size 100 --overall-wall-time 900
# """
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# -------- Defaults for your project --------
DEFAULT_PROJECT = r"C:\Users\Furqan\PycharmProjects\scenariogenerated"
DEFAULT_TESTS_ROOTS = [
    r"path_to_pynguin_generated_testcases\mbpp",
    r"path_to_pynguin_generated_testcases\humaneval",
]
DEFAULT_SOURCE_DIRS = [str(Path(DEFAULT_PROJECT) / "uut_pkg")]
DEFAULT_OUT_DIR = str(Path(DEFAULT_PROJECT) / "metrics_out")

# Windows-safe limit / chunking
CMD_LEN_SOFT_LIMIT = 24000  # be safe on Windows
DEFAULT_CHUNK_SIZE = 150    # nodeids per pytest run (auto-splits by length as well)


# -------- Helpers --------
def norm(p: str) -> str:
    try:
        return str(Path(p).resolve())
    except Exception:
        return os.path.abspath(p)


def discover_test_files(roots: List[str]) -> List[str]:
    patterns = ["**/pytest/test_*.py", "**/test_*.py", "**/*_test.py", "**/*tests.py"]
    files: List[str] = []
    for root in roots:
        base = Path(root)
        if not base.exists():
            continue
        for pat in patterns:
            files += [str(p) for p in base.rglob(pat)]
    # dedup preserve order
    seen, out = set(), []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    print(f"[INFO] Found {len(out)} test files under roots:")
    for r in roots:
        print("  -", r)
    for s in out[:8]:
        print("  example:", s)
    return out


def collect_nodeids(project_root: str, targets: List[str]) -> List[str]:
    """Collect test nodeids by pointing pytest at ROOTS (short command)."""
    cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q"] + targets
    proc = subprocess.run(cmd, cwd=project_root, text=True, capture_output=True)
    nodeids = []
    for line in (proc.stdout or "").splitlines():
        s = line.strip()
        if s and ("::" in s or s.endswith(".py")):
            nodeids.append(s)
    return nodeids


def find_coverage_files(search_roots: List[str]) -> List[str]:
    out = []
    for root in search_roots:
        p = Path(root)
        if not p.exists():
            continue
        out += [str(f) for f in p.rglob(".coverage*") if not f.name.endswith(".json")]
    # dedup
    out = sorted(set(out))
    return out


def cov_combine_and_json(project_root: str, in_files: List[str], out_dir: str) -> Optional[dict]:
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    cov_file = str(outp / ".coverage")
    cov_json = outp / "coverage.json"
    env = os.environ.copy()
    env["COVERAGE_FILE"] = cov_file
    if in_files:
        subprocess.run([sys.executable, "-m", "coverage", "combine"] + in_files,
                       cwd=project_root, env=env, check=False,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run([sys.executable, "-m", "coverage", "json", "-o", str(cov_json)],
                   cwd=project_root, env=env, check=False,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if cov_json.exists():
        try:
            return json.loads(cov_json.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def pytest_base_cmd(out_dir: str, source_dirs: List[str], branch: bool,
                    parallel: Optional[str], per_test_timeout: Optional[int]) -> Tuple[List[str], Dict[str, bool]]:
    have_xdist = False
    if parallel:
        try:
            import xdist  # type: ignore
            have_xdist = True
        except Exception:
            have_xdist = False
    have_timeout = False
    if per_test_timeout and per_test_timeout > 0:
        try:
            import pytest_timeout  # type: ignore
            have_timeout = True
        except Exception:
            have_timeout = False

    # Note: During chunked runs, JSON is produced at the very end by coverage json.
    cov_json = Path(out_dir) / "coverage.json"
    cmd = [sys.executable, "-m", "pytest", "--maxfail=0", "-q", "-p", "pytest_cov"]
    if branch:
        cmd += ["--cov-branch"]
    for src in source_dirs:
        cmd += [f"--cov={src}"]
    cmd += [f"--cov-report=json:{cov_json}"]

    if have_xdist:
        cmd += ["-p", "xdist.plugin", "-n", ("auto" if parallel == "auto" else str(parallel)), "--dist", "loadscope"]
    if have_timeout:
        cmd += ["-p", "pytest_timeout", "--timeout", str(per_test_timeout)]

    return cmd, {"xdist": have_xdist, "timeout": have_timeout}


def run_pytest_chunk(project_root: str, base_cmd: List[str], args_tail: List[str],
                     out_dir: str, append: bool, chunk_timeout: Optional[int]) -> Tuple[int, str, str]:
    """
    Run one pytest chunk. Uses COVERAGE_FILE in out_dir and --cov-append after the first chunk.
    Kills the chunk if it exceeds chunk_timeout seconds (returns rc=124) so the run can continue.
    """
    cmd = base_cmd[:]
    if append:
        cmd += ["--cov-append"]
    cmd += args_tail

    env = os.environ.copy()
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["COVERAGE_FILE"] = str(Path(out_dir) / ".coverage")

    try:
        proc = subprocess.run(
            cmd, cwd=project_root, text=True, capture_output=True, env=env,
            timeout=(chunk_timeout if chunk_timeout and chunk_timeout > 0 else None)
        )
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired as e:
        # 124 = conventional timeout rc
        msg = f"[WARN] Chunk timed out after {chunk_timeout}s; skipping remainder of this chunk.\n"
        return 124, msg + (e.stdout or ""), (e.stderr or "")


def chunkify_nodeids(nodeids: List[str], chunk_size: int) -> List[List[str]]:
    """
    Split nodeids into chunks, respecting command-length limit and requested chunk size.
    """
    chunks: List[List[str]] = []
    cur: List[str] = []
    cur_len = 0
    for nid in nodeids:
        add_len = len(nid) + 1
        # Split if adding would exceed Windows command-length safety
        if cur and (cur_len + add_len > CMD_LEN_SOFT_LIMIT):
            chunks.append(cur)
            cur = []
            cur_len = 0
        cur.append(nid)
        cur_len += add_len
        if chunk_size > 0 and len(cur) >= chunk_size:
            chunks.append(cur)
            cur = []
            cur_len = 0
    if cur:
        chunks.append(cur)
    return chunks


def parse_pytest_counts(stdout: str) -> Tuple[int, int]:
    failures = 0
    errors = 0
    text = " ".join(stdout.strip().splitlines())
    m = re.search(r"(\d+)\s+failed", text)
    if m:
        failures = int(m.group(1))
    m = re.search(r"(\d+)\s+errors?", text)
    if m:
        errors = int(m.group(1))
    return failures, errors


def ast_functions_in_file(pyfile: Path) -> List[Tuple[str, int, int]]:
    import ast
    try:
        src = pyfile.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    try:
        t = ast.parse(src)
    except Exception:
        return []
    out = []

    class V(ast.NodeVisitor):
        def __init__(self):
            self.scope = []

        def q(self, n):
            return ".".join(self.scope + [n]) if self.scope else n

        def visit_FunctionDef(self, n):
            out.append((self.q(n.name), getattr(n, "lineno", 1),
                        getattr(n, "end_lineno", getattr(n, "lineno", 1))))
            self.generic_visit(n)

        def visit_AsyncFunctionDef(self, n):
            out.append((self.q(n.name), getattr(n, "lineno", 1),
                        getattr(n, "end_lineno", getattr(n, "lineno", 1))))
            self.generic_visit(n)

        def visit_ClassDef(self, n):
            self.scope.append(n.name)
            self.generic_visit(n)
            self.scope.pop()

    V().visit(t)
    return out


def collect_source_files(source_dirs: List[str]) -> List[Path]:
    files: List[Path] = []
    for d in source_dirs:
        p = Path(d)
        if p.is_file() and p.suffix == ".py":
            files.append(p.resolve())
        elif p.is_dir():
            files += [q.resolve() for q in p.rglob("*.py")]
    # dedup
    seen = set()
    out: List[Path] = []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def coverage_metrics_from_json(cov_json: dict):
    files = cov_json.get("files", {})
    tot_stmts = tot_br = covered_lines = covered_br = 0
    executed_by_file: Dict[str, set] = {}
    summaries_by_file: Dict[str, Dict[str, int]] = {}
    for fp, info in files.items():
        summ = info.get("summary", {})
        ns = int(summ.get("num_statements", 0))
        cl = int(summ.get("covered_lines", 0))
        nb = int(summ.get("num_branches", 0))
        cb = int(summ.get("covered_branches", 0))
        tot_stmts += ns
        covered_lines += cl
        tot_br += nb
        covered_br += cb
        key = norm(fp)
        executed_by_file[key] = set(info.get("executed_lines", []) or [])
        summaries_by_file[key] = {
            "num_statements": ns,
            "covered_lines": cl,
            "num_branches": nb,
            "covered_branches": cb,
        }
    stmt = (covered_lines / tot_stmts * 100.0) if tot_stmts > 0 else 100.0
    br = (covered_br / tot_br * 100.0) if tot_br > 0 else 100.0
    path = br
    return stmt, br, path, executed_by_file, summaries_by_file


def function_and_depth_coverage(source_files: List[Path], executed_map: Dict[str, set]):
    total = covered = 0
    per_file: Dict[str, Tuple[float, float]] = {}
    for py in source_files:
        key = norm(str(py))
        executed = executed_map.get(key, set())
        funcs = ast_functions_in_file(py)
        if not funcs:
            continue
        f_total = f_covd = 0
        f_deep_vals: List[float] = []
        for _, start, end in funcs:
            total += 1
            f_total += 1
            lines = set(range(start, end + 1))
            hit = executed.intersection(lines)
            if hit:
                covered += 1
                f_covd += 1
                f_deep_vals.append(len(hit) / max(1, len(lines)) * 100.0)
        if f_total > 0:
            func_cov_file = (f_covd / f_total * 100.0)
            deep_cov_file = (sum(f_deep_vals) / len(f_deep_vals)) if f_deep_vals else 0.0
            per_file[key] = (func_cov_file, deep_cov_file)
    func_cov = (covered / total * 100.0) if total > 0 else 100.0
    deep_cov = (sum(v[1] for v in per_file.values()) / len(per_file)) if per_file else 0.0
    return func_cov, deep_cov, total, covered, per_file


def averages_from_per_file(summaries_by_file, per_file_func_deep):
    stmt_vals = []
    br_vals = []
    for s in summaries_by_file.values():
        ns = s.get("num_statements", 0)
        cl = s.get("covered_lines", 0)
        nb = s.get("num_branches", 0)
        cb = s.get("covered_branches", 0)
        if ns > 0:
            stmt_vals.append(cl / ns * 100.0)
        if nb > 0:
            br_vals.append(cb / nb * 100.0)
    func_vals = [v[0] for v in per_file_func_deep.values()]
    deep_vals = [v[1] for v in per_file_func_deep.values()]
    return {
        "avg_statement_percent_per_file": (sum(stmt_vals) / len(stmt_vals)) if stmt_vals else 0.0,
        "avg_branch_percent_per_file": (sum(br_vals) / len(br_vals)) if br_vals else 0.0,
        "avg_function_percent_per_file": (sum(func_vals) / len(func_vals)) if func_vals else 0.0,
        "avg_deep_percent_per_file": (sum(deep_vals) / len(deep_vals)) if deep_vals else 0.0,
    }


def compute_mi_for_sources(source_files: List[Path]) -> Optional[float]:
    try:
        from radon.metrics import mi_visit
    except Exception:
        return None
    vals = []
    for py in source_files:
        try:
            vals.append(mi_visit(py.read_text(encoding="utf-8", errors="ignore"), multi=True))
        except Exception:
            pass
    return (sum(vals) / len(vals)) if vals else None


def tce_from_mutation_results(path: Optional[str]):
    if not path:
        return None, None, 0, 0
    p = Path(path)
    if not p.exists():
        return None, None, 0, 0
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None, None, 0, 0
    killed = surv = to = err = inc = 0
    if isinstance(data, dict):
        killed += int(data.get("killed", 0))
        surv += int(data.get("survived", 0))
        to += int(data.get("timeouts", 0))
        err += int(data.get("errors", 0))
        inc += int(data.get("incompetent", 0))
        if isinstance(data.get("mutants"), list):
            for m in data["mutants"]:
                s = (m.get("status") or m.get("outcome") or "").lower()
                if s in {"killed", "detected"}:
                    killed += 1
                elif s == "survived":
                    surv += 1
                elif s == "timeout":
                    to += 1
                elif s == "incompetent":
                    inc += 1
                elif s == "error":
                    err += 1
    total = killed + surv + to + err + inc
    if total <= 0:
        return None, None, 0, 0
    tce = killed / total * 100.0
    ddp = tce
    false_alarms = err + inc
    return tce, ddp, killed, false_alarms


# -------- Main --------
def main():
    ap = argparse.ArgumentParser(description="Fast metrics with chunked pytest to avoid long commands.")
    ap.add_argument("--project-root", default=DEFAULT_PROJECT)
    ap.add_argument("--tests-roots", nargs="+", default=DEFAULT_TESTS_ROOTS)
    ap.add_argument("--source-dirs", nargs="+", default=DEFAULT_SOURCE_DIRS)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--mutation-results", default=None)

    # Fallback test-run knobs
    ap.add_argument("--parallel", default="auto", help="'auto' or an integer; '' to disable.")
    ap.add_argument("--per-test-timeout", type=int, default=2, help="Seconds per test (needs pytest-timeout). 0=off")
    ap.add_argument("--sample-per-file", type=int, default=-1, help="Max nodeids per file (-1=auto if big, 0=all)")
    ap.add_argument("--overall-sample", type=int, default=-1, help="Global cap (-1=auto if big, 0=all)")
    ap.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Nodeids per chunk (0=auto by length)")
    ap.add_argument("--branch", action="store_true", help="Enable branch coverage (slower). Default: statements only.")
    ap.add_argument("--overall-wall-time", type=int, default=0, help="Hard cap in seconds for running tests (0=off).")
    ap.add_argument("--chunk-timeout", type=int, default=120, help="Seconds per pytest CHUNK (kill and continue). 0=off")

    args = ap.parse_args()

    project_root = norm(args.project_root)
    tests_roots = [norm(p) for p in args.tests_roots]
    source_dirs = [norm(d) for d in args.source_dirs]
    out_dir = norm(args.out_dir)

    # Discovery
    test_files = discover_test_files(tests_roots)
    if not test_files:
        print(f"[ERROR] No test files found under: {tests_roots}")
        sys.exit(2)

    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    cov_json_path = outp / "coverage.json"

    # Fast path A: reuse JSON
    cov_json = None
    if cov_json_path.exists():
        try:
            cov_json = json.loads(cov_json_path.read_text(encoding="utf-8"))
            print(f"[INFO] Reused existing coverage: {cov_json_path}")
        except Exception:
            cov_json = None

    # Fast path B: combine existing .coverage* into JSON
    if cov_json is None:
        cov_files = find_coverage_files([project_root] + tests_roots + [out_dir])
        if cov_files:
            print(f"[INFO] Combining coverage files found: {len(cov_files)}")
            cov_json = cov_combine_and_json(project_root, cov_files, out_dir)
            if cov_json:
                print("[INFO] Combined -> coverage.json")

    failures = errors = 0

    # Fallback: run pytest in chunks
    if cov_json is None:
        all_nodes = collect_nodeids(project_root, tests_roots)
        if not all_nodes:
            print("[ERROR] No tests collected.")
            sys.exit(2)

        total_tests = len(all_nodes)
        by_file: Dict[str, List[str]] = {}
        for n in all_nodes:
            f = n.split("::", 1)[0]
            by_file.setdefault(f, []).append(n)

        # Auto-sample if huge; otherwise default to full suite
        sample_per_file = args.sample_per_file
        overall_sample = args.overall_sample
        if sample_per_file == -1 and overall_sample == -1:
            if total_tests > 3000:
                sample_per_file = 3
                overall_sample = 1000
            else:
                sample_per_file = 0
                overall_sample = 0

        nodeids: List[str] = []
        if sample_per_file != 0 or overall_sample != 0:
            for f, nodes in by_file.items():
                if sample_per_file > 0:
                    nodeids.extend(nodes[:sample_per_file])
                else:
                    nodeids.extend(nodes)
            if overall_sample > 0:
                nodeids = nodeids[:overall_sample]
            print(f"[INFO] Sampling {len(nodeids)}/{total_tests} tests "
                  f"(per-file={sample_per_file}, overall={overall_sample}).")
        else:
            nodeids = all_nodes
            print(f"[INFO] Running full suite: {len(nodeids)} tests.")

        base_cmd, features = pytest_base_cmd(out_dir, source_dirs, args.branch,
                                             args.parallel if args.parallel else None,
                                             args.per_test_timeout if args.per_test_timeout > 0 else None)

        chunks = chunkify_nodeids(nodeids, args.chunk_size)
        print(f"[INFO] Running in {len(chunks)} chunk(s), chunk_size={args.chunk_size} (auto split by length).")
        print(f"[INFO] Plugins active: xdist={features['xdist']}, timeout={features['timeout']}, branch_cov={args.branch}")

        start = time.monotonic()
        merged_cov_files: List[str] = []
        for i, ch in enumerate(chunks, 1):
            if args.overall_wall_time and (time.monotonic() - start) > args.overall_wall_time:
                print(f"[WARN] Overall wall-time cap reached ({args.overall_wall_time}s). Stopping early.")
                break

            rc, stdout, stderr = run_pytest_chunk(
                project_root, base_cmd, ch, out_dir, append=(i > 1), chunk_timeout=args.chunk_timeout
            )
            f_cnt, e_cnt = parse_pytest_counts(stdout)
            failures += f_cnt
            errors += e_cnt

            merged_cov_files = find_coverage_files([out_dir])

            if rc == 124:
                print(f"[INFO] Chunk {i}/{len(chunks)} TIMEOUT after {args.chunk_timeout}s; "
                      f"+failures={f_cnt}, +errors={e_cnt}, elapsed={(time.monotonic()-start):.1f}s, nodeids={len(ch)}")
            else:
                print(f"[INFO] Chunk {i}/{len(chunks)} done: rc={rc}, +failures={f_cnt}, +errors={e_cnt}, "
                      f"elapsed={(time.monotonic()-start):.1f}s, nodeids={len(ch)}")

        # After all chunks: combine and emit JSON
        cov_json = cov_combine_and_json(project_root, merged_cov_files, out_dir)
        if cov_json is None:
            print("[ERROR] Could not produce coverage.json after chunked runs.")
            sys.exit(2)
        print("[INFO] coverage.json written.")

    # ----- Compute metrics -----
    stmt_cov, br_cov, path_cov, executed_map, summaries_by_file = coverage_metrics_from_json(cov_json)
    source_files = collect_source_files(source_dirs)
    func_cov, deep_cov, total_funcs, covered_funcs, per_file_func_deep = function_and_depth_coverage(source_files, executed_map)
    shallow_cov = func_cov
    avgs = averages_from_per_file(summaries_by_file, per_file_func_deep)
    mi = compute_mi_for_sources(source_files)
    tce, ddp, killed_mutants, mut_false = tce_from_mutation_results(args.mutation_results)
    detected_bugs = failures + (killed_mutants if tce is not None else 0)
    false_alarms = errors + (mut_false if tce is not None else 0)

    # Print
    print(f"Function Coverage: {func_cov:.2f}%")
    print(f"Statement Coverage: {stmt_cov:.2f}%")
    print(f"Branch Coverage: {br_cov:.2f}%")
    print(f"Path Coverage: {path_cov:.2f}%")
    print(f"Shallow Coverage: {shallow_cov:.2f}%")
    print(f"Deep Coverage: {deep_cov:.2f}%")
    print(f"Test Case Effectiveness (TCE): {(tce if tce is not None else 0):.2f}%")
    print(f"Defect Detection Percentage (DDP): {(ddp if ddp is not None else 0):.2f}%")
    print(f"Maintainability Index: {mi if mi is not None else 'N/A'}")
    print(f"Number of Bugs Detected: {detected_bugs}")
    print(f"Number of False Alarms: {false_alarms}")
    print(f"Average Statement Coverage (per file): {avgs['avg_statement_percent_per_file']:.2f}%")
    print(f"Average Branch Coverage (per file): {avgs['avg_branch_percent_per_file']:.2f}%")
    print(f"Average Function Coverage (per file): {avgs['avg_function_percent_per_file']:.2f}%")
    print(f"Average Deep Coverage (per file): {avgs['avg_deep_percent_per_file']:.2f}%")

    # Save results
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "coverage.json").write_text(json.dumps({"files": cov_json.get("files", {})}, indent=2), encoding="utf-8")
    metrics = {
        "function_coverage_percent": func_cov,
        "statement_coverage_percent": stmt_cov,
        "branch_coverage_percent": br_cov,
        "path_coverage_percent": path_cov,
        "shallow_coverage_percent": shallow_cov,
        "deep_coverage_percent": deep_cov,
        "averages_per_file": avgs,
        "maintainability_index_avg": mi,
        "pytest_failures": failures,
        "pytest_errors": errors,
        "mutation": {
            "tce_percent": tce, "ddp_percent": ddp,
            "killed": killed_mutants if tce is not None else None,
            "false_alarms": mut_false if tce is not None else None,
            "results_file": args.mutation_results,
        },
        "functions": {"total": total_funcs, "covered": covered_funcs},
        "defaults_used": {
            "project_root": DEFAULT_PROJECT,
            "tests_roots": DEFAULT_TESTS_ROOTS,
            "source_dirs": DEFAULT_SOURCE_DIRS,
            "out_dir": DEFAULT_OUT_DIR,
        }
    }
    (Path(out_dir) / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    sys.exit(0)


if __name__ == "__main__":
    main()
