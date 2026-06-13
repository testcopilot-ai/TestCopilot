#!/usr/bin/env python3

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path


def read_text(path):
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_cmd(cmd, cwd=None, timeout=900):
    print("[CMD]", " ".join(str(x) for x in cmd))
    p = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    return p


def classify_fixed_result(returncode, stdout, stderr):
    combined = (stdout or "") + "\n" + (stderr or "")

    if returncode == 0:
        return "PASS_ON_FIXED"

    if (
        "can't open file" in combined
        or "No such file or directory" in combined
        or "ModuleNotFoundError" in combined
        or "ImportError" in combined
        or "ERROR conda.cli.main_run" in combined
    ):
        return "RUNTIME_ERROR_ON_FIXED"

    return "FAIL_ON_FIXED"


def load_official_patch(tasks_map_path, task_id):
    tasks_map = json.loads(read_text(tasks_map_path))
    task = tasks_map[task_id]

    for key in ["patch", "gold_patch", "fix_patch"]:
        patch = task.get(key, "")
        if isinstance(patch, str) and patch.strip():
            return patch

    raise RuntimeError(
        "No official patch found in tasks_map.json. "
        "Checked keys: patch, gold_patch, fix_patch."
    )


def copy_generated_test(test_file, fixed_repo, task_id):
    test_dir = fixed_repo / "tests" / "testcopilot_generated"
    test_dir.mkdir(parents=True, exist_ok=True)

    init_file = test_dir / "__init__.py"
    init_file.write_text("", encoding="utf-8")

    safe_task_id = re.sub(r"[^A-Za-z0-9_]+", "_", task_id)
    target_test = test_dir / f"test_generated_{safe_task_id}.py"

    test_code = read_text(test_file)
    write_text(target_test, test_code.strip() + "\n")

    return target_test


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task-dir",
        required=True,
        help="Path to real_project_tasks/swebench_django_11133",
    )

    parser.add_argument(
        "--validation-dir",
        required=True,
        help="Directory containing fixed_test_final.py from buggy validation",
    )

    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for fixed-version validation results",
    )

    parser.add_argument(
        "--test-file",
        default=None,
        help="Optional explicit generated test file. If omitted, uses validation-dir/fixed_test_final.py",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Timeout in seconds",
    )

    args = parser.parse_args()

    task_dir = Path(args.task_dir).expanduser()
    validation_dir = Path(args.validation_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = task_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.json: {metadata_path}")

    metadata = json.loads(read_text(metadata_path))

    task_id = metadata["task_id"]
    buggy_repo = Path(metadata["repo_path"]).expanduser()
    env_name = metadata.get("env_name", "setup_django__django__3.0")
    tasks_map_path = Path(metadata["tasks_map"]).expanduser()

    if args.test_file:
        test_file = Path(args.test_file).expanduser()
    else:
        test_file = validation_dir / "fixed_test_final.py"

    if not buggy_repo.exists():
        raise RuntimeError(f"Buggy repo does not exist: {buggy_repo}")

    if not (buggy_repo / "tests" / "runtests.py").exists():
        raise RuntimeError(
            f"Buggy repo is incomplete. Missing tests/runtests.py under: {buggy_repo}"
        )

    if not test_file.exists():
        raise FileNotFoundError(f"Generated test file not found: {test_file}")

    print("[INFO] Task:", task_id)
    print("[INFO] Buggy repo:", buggy_repo)
    print("[INFO] Conda env:", env_name)
    print("[INFO] Generated test:", test_file)

    # ---------------------------------------------------------
    # 1. Load official SWE-bench patch
    # ---------------------------------------------------------
    patch = load_official_patch(tasks_map_path, task_id)

    patch_file = out_dir / "official_fix_patch.diff"
    write_text(patch_file, patch)

    print("[INFO] Official patch saved to:", patch_file)

    # ---------------------------------------------------------
    # 2. Create fixed repo by copying buggy repo
    # ---------------------------------------------------------
    fixed_repo = out_dir / "fixed_repo"

    if fixed_repo.exists():
        shutil.rmtree(fixed_repo)

    print("[INFO] Copying buggy repo to fixed repo...")
    shutil.copytree(
        buggy_repo,
        fixed_repo,
        ignore=shutil.ignore_patterns(
            ".git",
            "__pycache__",
            "*.pyc",
            ".pytest_cache",
        ),
    )

    # Need git metadata for git apply? If .git ignored, git apply still works if run in worktree.
    # If git apply fails without .git, we fallback to patch -p1.
    print("[INFO] Applying official patch...")

    p_apply = run_cmd(["git", "apply", str(patch_file)], cwd=fixed_repo, timeout=args.timeout)

    write_text(out_dir / "patch_git_apply_stdout.txt", p_apply.stdout)
    write_text(out_dir / "patch_git_apply_stderr.txt", p_apply.stderr)

    if p_apply.returncode != 0:
        print("[WARN] git apply failed. Trying patch -p1...")
        p_patch = run_cmd(["patch", "-p1", "-i", str(patch_file)], cwd=fixed_repo, timeout=args.timeout)

        write_text(out_dir / "patch_p1_stdout.txt", p_patch.stdout)
        write_text(out_dir / "patch_p1_stderr.txt", p_patch.stderr)

        if p_patch.returncode != 0:
            summary = {
                "task_id": task_id,
                "status": "PATCH_APPLY_FAILED",
                "fixed_repo": str(fixed_repo),
                "patch_file": str(patch_file),
                "git_apply_returncode": p_apply.returncode,
                "patch_p1_returncode": p_patch.returncode,
                "git_apply_stderr": str(out_dir / "patch_git_apply_stderr.txt"),
                "patch_p1_stderr": str(out_dir / "patch_p1_stderr.txt"),
            }
            write_text(out_dir / "fixed_summary.json", json.dumps(summary, indent=2))
            print(json.dumps(summary, indent=2))
            raise RuntimeError("Failed to apply official patch.")

    # ---------------------------------------------------------
    # 3. Install fixed repo into task env
    # ---------------------------------------------------------
    print("[INFO] Installing fixed repo into conda env...")

    p_install = run_cmd(
        ["conda", "run", "-n", env_name, "python", "-m", "pip", "install", "-e", str(fixed_repo)],
        cwd=fixed_repo,
        timeout=args.timeout,
    )

    write_text(out_dir / "fixed_install_stdout.txt", p_install.stdout)
    write_text(out_dir / "fixed_install_stderr.txt", p_install.stderr)

    if p_install.returncode != 0:
        summary = {
            "task_id": task_id,
            "status": "FIXED_INSTALL_FAILED",
            "fixed_repo": str(fixed_repo),
            "install_returncode": p_install.returncode,
            "install_stdout": str(out_dir / "fixed_install_stdout.txt"),
            "install_stderr": str(out_dir / "fixed_install_stderr.txt"),
        }
        write_text(out_dir / "fixed_summary.json", json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        raise RuntimeError("Failed to install fixed repo.")

    # ---------------------------------------------------------
    # 4. Copy generated test into fixed repo
    # ---------------------------------------------------------
    copied_test = copy_generated_test(test_file, fixed_repo, task_id)
    print("[INFO] Copied generated test to:", copied_test)

    # ---------------------------------------------------------
    # 5. Run generated test on fixed version
    # ---------------------------------------------------------
    print("[INFO] Running generated test on fixed version...")

    cmd = [
        "conda",
        "run",
        "-n",
        env_name,
        "python",
        "tests/runtests.py",
        "testcopilot_generated",
        "-v",
        "2",
    ]

    p_test = run_cmd(cmd, cwd=fixed_repo, timeout=args.timeout)

    write_text(out_dir / "fixed_stdout.txt", p_test.stdout)
    write_text(out_dir / "fixed_stderr.txt", p_test.stderr)

    status = classify_fixed_result(p_test.returncode, p_test.stdout, p_test.stderr)

    summary = {
        "task_id": task_id,
        "project": metadata.get("project", "Django"),
        "env_name": env_name,
        "fixed_repo": str(fixed_repo),
        "generated_test": str(copied_test),
        "returncode": p_test.returncode,
        "status": status,
        "stdout": str(out_dir / "fixed_stdout.txt"),
        "stderr": str(out_dir / "fixed_stderr.txt"),
        "patch_file": str(patch_file),
    }

    write_text(out_dir / "fixed_summary.json", json.dumps(summary, indent=2))

    print("\nDONE")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
