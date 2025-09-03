#!/usr/bin/env python
# -*- coding: utf-8 -*-
# """
# mutation_metrics.py — mutate code, then compute:
#   - False Alarms (tests failing on correct code)
#   - Bugs Detected (killed mutants)
#   - TCE count (tests passing on correct, failing on buggy)
#   - TCE% (unique effective tests / baseline passed)
#   - DDP% (killed mutants / total mutants)
#
# Fast & Windows-safe:
#   - Runs pytest in CHUNKS (avoids long cmd lines)
#   - Optional parallelism (pytest-xdist)
#   - Optional per-test timeouts (pytest-timeout)
#   - Optional sampling and overall wall-time cap
#
# Requires:
#   pip install pytest
# Optional (recommended):
#   pip install pytest-xdist pytest-timeout
# """
import argparse, os, sys, subprocess, shutil, time, xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# ---- Defaults for YOUR project ----
DEFAULT_PROJECT = r"directory_to_dataset"
DEFAULT_TESTS_ROOTS = [
    r"Path_to_pyuguin_generated_test_cases\pynguin_results\mbpp",
    r"Path_to_pyuguin_generated_test_cases\pynguin_results\humaneval",
]
DEFAULT_SOURCE_DIR = str(Path(DEFAULT_PROJECT) / "uut_pkg")
DEFAULT_OUT_DIR = str(Path(DEFAULT_PROJECT) / "mutation_out")

CMD_LEN_SOFT_LIMIT = 24000
DEFAULT_CHUNK_SIZE = 120  # nodeids per pytest run
DEFAULT_MUTANTS_PER_FILE = 5
DEFAULT_MAX_MUTANTS = 40

# ----------------- Pytest helpers -----------------
def norm(p: str) -> str:
    try: return str(Path(p).resolve())
    except Exception: return str(Path(p))

def discover_test_files(roots: List[str]) -> List[str]:
    patt = ["**/pytest/test_*.py", "**/test_*.py", "**/*_test.py", "**/*tests.py"]
    files: List[str] = []
    for root in roots:
        base = Path(root)
        if not base.exists(): continue
        for pat in patt:
            files += [str(p) for p in base.rglob(pat)]
    # de-dup preserve order
    seen, out = set(), []
    for f in files:
        if f not in seen:
            seen.add(f); out.append(f)
    print(f"[INFO] Found {len(out)} test files.")
    for s in out[:8]: print("  example:", s)
    return out

def collect_nodeids(project_root: str, targets: List[str]) -> List[str]:
    cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q"] + targets
    proc = subprocess.run(cmd, cwd=project_root, text=True, capture_output=True)
    nodeids=[]
    for line in (proc.stdout or "").splitlines():
        s=line.strip()
        if s and ("::" in s or s.endswith(".py")):
            nodeids.append(s)
    return nodeids

def chunkify(items: List[str], chunk_size: int) -> List[List[str]]:
    if chunk_size <= 0: chunk_size = DEFAULT_CHUNK_SIZE
    chunks=[]; cur=[]; cur_len=0
    for it in items:
        add_len=len(it)+1
        if cur and (cur_len+add_len > CMD_LEN_SOFT_LIMIT):
            chunks.append(cur); cur=[]; cur_len=0
        cur.append(it); cur_len += add_len
        if len(cur) >= chunk_size:
            chunks.append(cur); cur=[]; cur_len=0
    if cur: chunks.append(cur)
    return chunks

def pytest_base_cmd(parallel: Optional[str], per_test_timeout: Optional[int]) -> Tuple[List[str], Dict[str,bool]]:
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
    cmd = [sys.executable, "-m", "pytest", "--maxfail=0", "-q"]
    if have_xdist:
        cmd += ["-p","xdist.plugin","-n", ("auto" if parallel=="auto" else str(parallel)), "--dist","loadscope"]
    if have_timeout:
        cmd += ["-p","pytest_timeout","--timeout", str(per_test_timeout)]
    return cmd, {"xdist": have_xdist, "timeout": have_timeout}

def run_pytest_chunk(project_root: str, base_cmd: List[str], nodeids: List[str],
                     junit_path: Path, extra_env: Optional[Dict[str,str]]=None,
                     chunk_timeout: Optional[int]=None) -> Tuple[int,str,str]:
    cmd = base_cmd[:] + ["--junitxml", str(junit_path)] + nodeids
    env = os.environ.copy()
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if extra_env:
        env.update(extra_env)
    try:
        proc = subprocess.run(cmd, cwd=project_root, text=True, capture_output=True,
                              timeout=(chunk_timeout if chunk_timeout and chunk_timeout>0 else None),
                              env=env)
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired as e:
        # 124 = timeout
        with open(junit_path, "w", encoding="utf-8") as f:
            f.write("<testsuite name='timeout' tests='0'></testsuite>")
        return 124, (e.stdout or ""), (e.stderr or "")

def parse_junit_tests(junit_files: List[Path]) -> Dict[str,str]:
    """
    Return map test_id -> status ('passed'|'failed'|'error'|'skipped').
    test_id is canonicalized as: (file or classname) + '::' + name
    """
    results: Dict[str,str] = {}
    for jf in junit_files:
        if not jf.exists(): continue
        try:
            root = ET.fromstring(jf.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        # JUnit may have <testsuite> root or <testsuites> wrapper
        suites = []
        if root.tag.endswith("testsuite"):
            suites = [root]
        else:
            suites = [ts for ts in root.iter() if ts.tag.endswith("testsuite")]
        for ts in suites:
            for tc in ts.findall(".//testcase"):
                name = tc.attrib.get("name","")
                file_attr = tc.attrib.get("file","")
                classname = tc.attrib.get("classname","")
                left = file_attr or classname or "<unknown>"
                tid = f"{left}::{name}"
                status = "passed"
                if tc.find("failure") is not None:
                    status = "failed"
                elif tc.find("error") is not None:
                    status = "error"
                elif tc.find("skipped") is not None:
                    status = "skipped"
                results[tid] = status
    return results

# ----------------- Mutant generation (simple AST) -----------------
import ast

class SingleMutation(ast.NodeTransformer):
    """
    Applies exactly ONE mutation at target index among discovered candidates.
    """
    def __init__(self, target_index: int):
        self.target_index = target_index
        self.counter = 0
        super().__init__()

    def mutate(self, node):
        self.counter += 1
        if self.counter-1 == self.target_index:
            return True
        return False

    # Binary ops (+ - * / // %)
    def visit_BinOp(self, node: ast.BinOp):
        self.generic_visit(node)
        if self.mutate(node):
            swap = {
                ast.Add: ast.Sub,
                ast.Sub: ast.Add,
                ast.Mult: ast.Add,
                ast.Div: ast.Mult,
                ast.FloorDiv: ast.Mult,
                ast.Mod: ast.Add,
            }
            op_type = type(node.op)
            new_op = swap.get(op_type)
            if new_op:
                return ast.copy_location(ast.BinOp(left=node.left, op=new_op(), right=node.right), node)
        return node

    # Comparisons (<, <=, >, >=, ==, !=)
    def visit_Compare(self, node: ast.Compare):
        self.generic_visit(node)
        if self.mutate(node) and node.ops:
            op = node.ops[0]
            swap = {
                ast.Lt: ast.Gt, ast.Gt: ast.Lt,
                ast.LtE: ast.GtE, ast.GtE: ast.LtE,
                ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
            }
            op_type = type(op)
            new_op = swap.get(op_type)
            if new_op:
                node.ops[0] = new_op()
        return node

    # Unary ops: +x -> -x, -x -> +x, not x -> x
    def visit_UnaryOp(self, node: ast.UnaryOp):
        self.generic_visit(node)
        if self.mutate(node):
            if isinstance(node.op, ast.UAdd):
                return ast.copy_location(ast.UnaryOp(op=ast.USub(), operand=node.operand), node)
            if isinstance(node.op, ast.USub):
                return ast.copy_location(ast.UnaryOp(op=ast.UAdd(), operand=node.operand), node)
            if isinstance(node.op, ast.Not):
                return node.operand
        return node

    # Constants: int n -> n+1
    def visit_Constant(self, node: ast.Constant):
        self.generic_visit(node)
        if self.mutate(node):
            if isinstance(node.value, bool):
                return ast.copy_location(ast.Constant(value=not node.value), node)
            if isinstance(node.value, int):
                return ast.copy_location(ast.Constant(value=node.value + 1), node)
        return node

def enumerate_candidates(tree: ast.AST) -> int:
    """
    Count how many nodes are potential mutation points (the same as SingleMutation visits).
    """
    class Counter(ast.NodeVisitor):
        def __init__(self): self.count=0
        def visit_BinOp(self, n): self.count+=1; self.generic_visit(n)
        def visit_Compare(self, n): self.count+=1; self.generic_visit(n)
        def visit_UnaryOp(self, n): self.count+=1; self.generic_visit(n)
        def visit_Constant(self, n): self.count+=1; self.generic_visit(n)
    c=Counter(); c.visit(tree); return c.count

def make_mutants_for_file(pyfile: Path, max_per_file: int) -> List[Tuple[str, str]]:
    """
    Returns list of (label, mutated_source) for up to max_per_file mutants.
    """
    try:
        src = pyfile.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    try:
        tree = ast.parse(src)
    except Exception:
        return []

    total = enumerate_candidates(tree)
    mutants=[]
    for idx in range(total):
        if len(mutants) >= max_per_file:
            break
        mut = SingleMutation(idx)
        new_tree = mut.visit(ast.fix_missing_locations(ast.parse(src)))
        try:
            mutated_src = ast.unparse(new_tree)  # Python 3.9+
        except Exception:
            try:
                import astor  # type: ignore
                mutated_src = astor.to_source(new_tree)
            except Exception:
                continue
        mutants.append((f"mut_{pyfile.stem}_{idx}", mutated_src))
    return mutants

def collect_source_py_files(source_dir: str) -> List[Path]:
    p = Path(source_dir)
    if p.is_file() and p.suffix==".py": return [p.resolve()]
    return [q.resolve() for q in Path(source_dir).rglob("*.py")]

# ----------------- Main workflow -----------------
def main():
    ap = argparse.ArgumentParser(description="Mutation metrics: False Alarms, Bugs Detected, TCE, DDP.")
    ap.add_argument("--project-root", default=DEFAULT_PROJECT)
    ap.add_argument("--tests-roots", nargs="+", default=DEFAULT_TESTS_ROOTS)
    ap.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)

    # Test running controls
    ap.add_argument("--parallel", default="auto", help="'auto' or an integer; '' to disable.")
    ap.add_argument("--per-test-timeout", type=int, default=2, help="pytest-timeout seconds per test (0=off)")
    ap.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    ap.add_argument("--chunk-timeout", type=int, default=120, help="sec per pytest chunk (kills chunk if exceeded, 0=off)")
    ap.add_argument("--overall-wall-time", type=int, default=0, help="overall cap in sec for baseline or per-mutant run (0=off)")
    ap.add_argument("--sample-per-file", type=int, default=2, help="max tests per file for speed (0=all)")
    ap.add_argument("--overall-sample", type=int, default=800, help="global cap on tests (0=all)")

    # Mutation controls
    ap.add_argument("--max-mutants", type=int, default=DEFAULT_MAX_MUTANTS)
    ap.add_argument("--mutants-per-file", type=int, default=DEFAULT_MUTANTS_PER_FILE)

    args = ap.parse_args()

    project_root = norm(args.project_root)
    tests_roots = [norm(r) for r in args.tests_roots]
    source_dir = norm(args.source_dir)
    out_dir = norm(args.out_dir)
    outp = Path(out_dir); outp.mkdir(parents=True, exist_ok=True)

    # 1) Discover & sample tests
    test_files = discover_test_files(tests_roots)
    if not test_files:
        print("[ERROR] No tests found.")
        sys.exit(2)
    all_nodes = collect_nodeids(project_root, tests_roots)
    if not all_nodes:
        print("[ERROR] No tests collected.")
        sys.exit(2)

    # Group by file for sampling
    by_file: Dict[str, List[str]] = {}
    for n in all_nodes:
        f = n.split("::", 1)[0]
        by_file.setdefault(f, []).append(n)

    nodeids: List[str] = []
    for f, nodes in by_file.items():
        if args.sample_per_file > 0:
            nodeids.extend(nodes[:args.sample_per_file])
        else:
            nodeids.extend(nodes)
    if args.overall_sample > 0:
        nodeids = nodeids[:args.overall_sample]

    print(f"[INFO] Using {len(nodeids)} tests for metrics (sample-per-file={args.sample_per_file}, overall={args.overall_sample}).")

    # 2) BASELINE run (correct code)
    base_cmd, features = pytest_base_cmd(args.parallel if args.parallel else None,
                                         args.per_test_timeout if args.per_test_timeout>0 else None)
    chunks = chunkify(nodeids, args.chunk_size)
    print(f"[INFO] Baseline in {len(chunks)} chunk(s). Plugins: xdist={features['xdist']}, timeout={features['timeout']}")
    baseline_xmls = []
    t0=time.monotonic()
    for i,ch in enumerate(chunks,1):
        if args.overall_wall_time and (time.monotonic()-t0) > args.overall_wall_time:
            print(f"[WARN] Baseline wall-time cap reached ({args.overall_wall_time}s). Stopping early.")
            break
        junit_path = Path(out_dir) / f"baseline_chunk_{i}.xml"
        rc, so, se = run_pytest_chunk(project_root, base_cmd, ch, junit_path,
                                      extra_env=None, chunk_timeout=args.chunk_timeout)
        baseline_xmls.append(junit_path)
        print(f"[INFO] Baseline chunk {i}/{len(chunks)} rc={rc} tests={len(ch)}")

    baseline_results = parse_junit_tests(baseline_xmls)
    baseline_failed = {tid for tid,st in baseline_results.items() if st in {"failed","error"}}
    baseline_passed = {tid for tid,st in baseline_results.items() if st=="passed"}
    print(f"[INFO] Baseline: passed={len(baseline_passed)}, failed={len(baseline_failed)}")

    false_alarms = len(baseline_failed)  # as per your definition

    # 3) Build mutants (limited)
    py_files = collect_source_py_files(source_dir)
    mutants: List[Tuple[Path, str, str]] = []  # (source_file, label, mutated_src)
    for py in py_files:
        file_mutants = make_mutants_for_file(py, args.mutants_per_file)
        for (label, msrc) in file_mutants:
            mutants.append((py, label, msrc))
            if len(mutants) >= args.max_mutants:
                break
        if len(mutants) >= args.max_mutants:
            break
    if not mutants:
        print("[WARN] No mutants generated (no candidates).")
        print(f"False Alarms: {false_alarms}")
        print("Bugs Detected: 0\nTCE (count): 0\nTCE (%): 0.00%\nDDP (%): 0.00%")
        sys.exit(0)
    print(f"[INFO] Generated {len(mutants)} mutants (cap per file={args.mutants_per_file}, total cap={args.max_mutants}).")

    # 4) For each mutant: write to mut_work and run sampled tests
    killed_mutants = 0
    tce_union_tests = set()

    for idx, (src_file, label, msrc) in enumerate(mutants, 1):
        mut_root = Path(out_dir) / f"mut_work_{idx}"
        if mut_root.exists(): shutil.rmtree(mut_root)
        mut_root.mkdir(parents=True, exist_ok=True)

        # replicate package structure: put mutated file into mut_root at same relative path
        src_path = Path(source_dir)
        rel = Path(src_file).relative_to(src_path)
        target_file = mut_root / rel
        target_file.parent.mkdir(parents=True, exist_ok=True)

        # copy all original sources first
        for f in collect_source_py_files(source_dir):
            relf = Path(f).relative_to(src_path)
            dst = mut_root / relf
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)

        # overwrite the chosen file with mutated source
        target_file.write_text(msrc, encoding="utf-8")

        # run tests against mutated code; prepend mut_root's parent to PYTHONPATH so 'uut_pkg' resolves here first
        junit_files=[]
        chunks_m = chunkify(nodeids, args.chunk_size)
        start_mut = time.monotonic()
        for ci, ch in enumerate(chunks_m, 1):
            if args.overall_wall_time and (time.monotonic()-start_mut) > args.overall_wall_time:
                print(f"[WARN] Mutant {idx} wall-time cap reached ({args.overall_wall_time}s).")
                break
            junit_path = Path(out_dir) / f"{label}_chunk_{ci}.xml"
            extra_env = {"PYTHONPATH": f"{mut_root.parent}{os.pathsep}{os.environ.get('PYTHONPATH','')}"}
            rc, so, se = run_pytest_chunk(project_root, base_cmd, ch, junit_path,
                                          extra_env=extra_env, chunk_timeout=args.chunk_timeout)
            junit_files.append(junit_path)

        mut_results = parse_junit_tests(junit_files)
        mut_failed = {tid for tid,st in mut_results.items() if st in {"failed","error"}}
        effective_tests = baseline_passed.intersection(mut_failed)

        if effective_tests:
            killed_mutants += 1
            tce_union_tests.update(effective_tests)

        print(f"[INFO] Mutant {idx}/{len(mutants)} {label}: killed={bool(effective_tests)} "
              f"(effective_tests={len(effective_tests)})")

        # cleanup work dir to save space
        shutil.rmtree(mut_root, ignore_errors=True)

    # 5) Metrics
    total_mutants = len(mutants)
    bugs_detected = killed_mutants
    tce_count = len(tce_union_tests)
    tce_percent = (tce_count / max(1, len(baseline_passed))) * 100.0
    ddp_percent = (killed_mutants / max(1, total_mutants)) * 100.0

    print("\n=== Mutation Metrics ===")
    print(f"False Alarms (baseline failed tests): {false_alarms}")
    print(f"Bugs Detected (killed mutants): {bugs_detected} / {total_mutants}")
    print(f"TCE (count, unique tests): {tce_count}")
    print(f"TCE (% of baseline passed): {tce_percent:.2f}%")
    print(f"DDP (% killed mutants): {ddp_percent:.2f}%")

if __name__ == "__main__":
    main()
