# """
# generate_pynguin_tests_hardcoded_sanitizing.py
#
# Hard-coded runner for your two datasets with a SANITIZER that auto-fixes
# common syntax glitches in dataset code cells:
#
#   - Lines like "return X:"  -> "return X"
#   - "return:"               -> "return"
#   - "raise X:"              -> "raise X"
#   - "break:" / "continue:" / "pass:" -> without trailing colon
#
# If code still doesn't compile, it iteratively COMMENTS the offending line
# (based on SyntaxError.lineno) up to 5 times so the module becomes importable.
#
# Datasets (hard-coded):
#  Path to dataset
#
# Output:
#   <project>\pynguin_results\<dataset_tag>\... with a summary.csv including:
#   row_idx, module_name, status, return_code, sanitized, commented, error
#
# Usage (in your venv):
#   python generate_pynguin_tests_hardcoded_sanitizing.py --clean --rows 0:50 --timeout 180
# """
import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import pandas as pd

# -----------------------
# Configuration defaults
# -----------------------
PROJECT_PATH = r"path_to_dataset"

# Hard-coded datasets
DATASETS = [
    {
        "tag": "mbpp",
        "path": Path(PROJECT_PATH) / "mbpp_test_cases.xlsx",
        "default_sheet": "Sheet1",
    },
    {
        "tag": "humaneval",
        "path": Path(PROJECT_PATH) / "TestcasesHumanEvalu.xlsx",
        "default_sheet": "Sheet2",
    },
]

# -----------------------
# Helpers
# -----------------------
def extract_function_name(signature: str) -> Optional[str]:
    m = re.match(r"\s*def\s+([A-Za-z_]\w*)\s*\(", signature or "")
    return m.group(1) if m else None


def parse_rows_arg(rows_arg: Optional[str], max_index: int) -> range:
    if not rows_arg:
        return range(0, max_index + 1)
    if ":" in rows_arg:
        a, b = rows_arg.split(":", 1)
        start = int(a) if a else 0
        end = int(b) if b else (max_index + 1)
        return range(start, min(end, max_index + 1))
    i = int(rows_arg)
    return range(i, i + 1)


def pick_sheet(df_or_dict: Any, preferred: Optional[str]) -> Tuple[pd.DataFrame, str]:
    """Return (DataFrame, sheet_name) for either DataFrame or dict-of-DataFrames."""
    if isinstance(df_or_dict, dict):
        sheets: Dict[str, pd.DataFrame] = df_or_dict
        if preferred and preferred in sheets:
            return sheets[preferred], preferred
        first_name = next(iter(sheets.keys()))
        return sheets[first_name], first_name
    return df_or_dict, preferred or "<first-sheet>"


def autodetect_columns(df: pd.DataFrame, code_col_arg: Optional[str], sig_col_arg: Optional[str]) -> Tuple[str, str]:
    if code_col_arg and code_col_arg in df.columns:
        code_col = code_col_arg
    else:
        code_col = next((c for c in ("code", "Code") if c in df.columns), None)

    if sig_col_arg and sig_col_arg in df.columns:
        sig_col = sig_col_arg
    else:
        sig_col = next((s for s in ("signatures", "signature", "Signature") if s in df.columns), None)

    if not code_col or not sig_col:
        raise SystemExit(f"[ERROR] Could not find required columns. Found columns: {list(df.columns)}")
    return code_col, sig_col


# --- Sanitizer ---
_re_return_expr_colon = re.compile(r'^(\s*)(return\s+.+):\s*$')
_re_return_only_colon = re.compile(r'^(\s*)(return)\s*:\s*$')
_re_raise_expr_colon  = re.compile(r'^(\s*)(raise\s+.+):\s*$')
_re_simple_kw_colon   = re.compile(r'^(\s*)(break|continue|pass)\s*:\s*$')

def _remove_trailing_colon_typos(line: str) -> str:
    # Only touch lines that end with a colon but start with these statements.
    m = _re_return_expr_colon.match(line)
    if m:
        return f"{m.group(1)}{m.group(2)}"
    m = _re_return_only_colon.match(line)
    if m:
        return f"{m.group(1)}{m.group(2)}"
    m = _re_raise_expr_colon.match(line)
    if m:
        return f"{m.group(1)}{m.group(2)}"
    m = _re_simple_kw_colon.match(line)
    if m:
        return f"{m.group(1)}{m.group(2)}"
    return line


def sanitize_code_once(src: str) -> str:
    # Normalize newlines and fix the known patterns per line
    lines = src.replace('\r\n', '\n').replace('\r', '\n').split('\n')
    fixed = [_remove_trailing_colon_typos(l) for l in lines]
    return '\n'.join(fixed)


def try_compile(src: str, name: str = "<module>") -> Tuple[bool, Optional[BaseException]]:
    try:
        compile(src, name, "exec")
        return True, None
    except BaseException as e:
        return False, e


def sanitize_until_compiles(src: str, max_comment_passes: int = 5) -> Tuple[str, bool, int, Optional[str]]:
    """
    Returns (final_src, sanitized_bool, commented_count, last_error_str).
    1) First pass removes trailing colon typos on return/raise/break/continue/pass.
    2) If still failing, iteratively comments the offending line up to N passes.
    """
    sanitized = False
    commented = 0

    ok, err = try_compile(src)
    if ok:
        return src, False, 0, None

    # Try pattern-based typo fixes
    src2 = sanitize_code_once(src)
    if src2 != src:
        sanitized = True
    ok, err = try_compile(src2)
    if ok:
        return src2, sanitized, 0, None

    # Iteratively comment the offending line (syntax errors)
    # Only handle SyntaxError with .lineno
    current = src2
    for _ in range(max_comment_passes):
        ok, err = try_compile(current)
        if ok:
            return current, sanitized, commented, None
        if isinstance(err, SyntaxError) and getattr(err, "lineno", None):
            lines = current.splitlines()
            lineno = max(1, min(len(lines), err.lineno))
            original = lines[lineno - 1]
            # Avoid re-commenting already commented lines
            if not original.lstrip().startswith("#"):
                lines[lineno - 1] = f"# [SANITIZED: was invalid] {original}"
                commented += 1
                sanitized = True
                current = "\n".join(lines)
            else:
                # If it's already a comment, break to avoid infinite loop
                break
        else:
            # Non-syntax exception during compile: stop
            break

    # Final attempt
    ok, err = try_compile(current)
    return (current, sanitized, commented, repr(err) if err else None)


# --- Pynguin runner (with DANGER var set) ---
def run_pynguin(project_path: Path, module_name: str, outdir: Path, timeout: Optional[int]) -> int:
    """
    Invoke Pynguin CLI with DANGER flag set so it won't abort.
    """
    env = os.environ.copy()
    # This value can be anything; Pynguin only checks that the var exists.
    env["PYNGUIN_DANGER_AWARE"] = "1"

    cmd = [
        sys.executable.replace("pythonw.exe", "python.exe"),
        "-m",
        "pynguin",
        "--project-path", str(project_path),
        "--module-name", module_name,
        "--output-path", str(outdir),
        "-v",
    ]
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            stdout=sys.stdout,
            stderr=sys.stderr,
            timeout=timeout,
            env=env,
        )
        return proc.returncode
    except subprocess.TimeoutExpired:
        print(f"[WARN] Pynguin timed out for module '{module_name}'.")
        return 124


# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser(description="Run Pynguin over two hard-coded datasets with code sanitization.")
    ap.add_argument("--project-path", default=PROJECT_PATH, help="Root of your project (imports, generated package).")
    ap.add_argument("--outdir", default=None, help="Output root (default: <project>\\pynguin_results).")
    ap.add_argument("--mbpp-sheet", default=None, help="Override sheet for mbpp_test_cases.xlsx (default: Sheet1).")
    ap.add_argument("--humaneval-sheet", default=None, help="Override sheet for TestcasesHumanEvalu.xlsx (default: Sheet2).")
    ap.add_argument("--code-col", default=None, help="Column containing source code (auto: code/Code).")
    ap.add_argument("--sig-col", default=None, help="Column containing signature (auto: signatures/signature/Signature).")
    ap.add_argument("--package-name", default="uut_pkg", help="Package name to write modules into (under project).")
    ap.add_argument("--rows", default=None, help="Row selection e.g. 0:50, 10:, 7 (applied to BOTH datasets).")
    ap.add_argument("--timeout", type=int, default=None, help="Seconds per module for Pynguin.")
    ap.add_argument("--clean", action="store_true", help="Clean output directory before generating.")
    args = ap.parse_args()

    project_path = Path(args.project_path).resolve()
    if not project_path.exists():
        print(f"[ERROR] Project path does not exist: {project_path}")
        sys.exit(2)

    out_root = Path(args.outdir) if args.outdir else (project_path / "pynguin_results")
    if args.clean and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Prepare/ensure target package for modules
    pkg_dir = project_path / args.package_name
    pkg_dir.mkdir(parents=True, exist_ok=True)
    init_file = pkg_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text("# auto-generated package for Pynguin targets\n", encoding="utf-8")

    # Process both datasets
    for ds in DATASETS:
        tag = ds["tag"]
        xlsx_path: Path = ds["path"]
        if not xlsx_path.exists():
            print(f"[WARN] Dataset not found, skipping: {xlsx_path}")
            continue

        # Sheet selection per dataset
        preferred_sheet = None
        if tag == "mbpp":
            preferred_sheet = args.mbpp_sheet or ds["default_sheet"]
        elif tag == "humaneval":
            preferred_sheet = args.humaneval_sheet or ds["default_sheet"]

        print(f"[INFO] Loading dataset '{tag}' from: {xlsx_path}")
        if preferred_sheet:
            df_or_dict = pd.read_excel(xlsx_path, sheet_name=preferred_sheet)
        else:
            df_or_dict = pd.read_excel(xlsx_path, sheet_name=None)
        df, sheet_used = pick_sheet(df_or_dict, preferred_sheet)
        print(f"[INFO] Using sheet for {tag}: {sheet_used}")

        # Autodetect columns
        code_col, sig_col = autodetect_columns(df, args.code_col, args.sig_col)
        print(f"[INFO] Using columns for {tag} -> code: '{code_col}', signature: '{sig_col}'")

        # Row range
        max_index = len(df) - 1
        row_range = parse_rows_arg(args.rows, max_index)

        # Per-dataset output
        ds_outdir = out_root / tag
        ds_outdir.mkdir(parents=True, exist_ok=True)

        summary_lines: List[str] = ["row_idx,module_name,status,return_code,sanitized,commented,error"]
        processed = 0

        for idx in row_range:
            row = df.iloc[idx]
            raw_code = str(row[code_col]) if pd.notna(row[code_col]) else ""
            signature = str(row[sig_col]) if pd.notna(row[sig_col]) else ""

            if not raw_code.strip() or not signature.strip():
                print(f"[SKIP] {tag} row {idx}: missing code or signature")
                summary_lines.append(f"{idx},,skipped_missing,,False,0,")
                continue

            fn_name = extract_function_name(signature)
            if not fn_name:
                print(f"[SKIP] {tag} row {idx}: could not parse function name from signature: {signature!r}")
                summary_lines.append(f"{idx},,skipped_bad_signature,,False,0,")
                continue

            # Sanitize & ensure compiles
            sanitized_src, did_sanitize, commented_count, last_err = sanitize_until_compiles(
                raw_code, max_comment_passes=5
            )

            # If still doesn't compile, skip
            ok, _ = try_compile(sanitized_src)
            if not ok:
                print(f"[SKIP] {tag} row {idx}: code remains uncompilable after sanitization.")
                summary_lines.append(f"{idx},,skipped_uncompilable,,{did_sanitize},{commented_count},{last_err}")
                continue

            module_name = f"{tag}_row_{idx}_{fn_name}"
            module_path = pkg_dir / f"{module_name}.py"
            module_path.write_text(sanitized_src, encoding="utf-8")

            module_out = ds_outdir / module_name
            module_out.mkdir(parents=True, exist_ok=True)

            fqmn = f"{args.package_name}.{module_name}"
            print(f"[INFO] Running Pynguin for {fqmn}")
            rc = run_pynguin(project_path=project_path, module_name=fqmn, outdir=module_out, timeout=args.timeout)

            status = "ok" if rc == 0 else f"error_{rc}"
            summary_lines.append(f"{idx},{fqmn},{status},{rc},{did_sanitize},{commented_count},")

            processed += 1

        (ds_outdir / "summary.csv").write_text("\n".join(summary_lines), encoding="utf-8")
        print(f"[DONE] {tag}: processed {processed} rows. See {ds_outdir/'summary.csv'}.")

    print(f"[ALL DONE] Results under: {out_root}")
    print(f"[NOTE] Generated tests are under {out_root}\\<dataset_tag>\\<module_name>\\.")
    print(f"[NOTE] summary.csv columns: row_idx, module_name, status, return_code, sanitized, commented, error")


if __name__ == "__main__":
    main()
