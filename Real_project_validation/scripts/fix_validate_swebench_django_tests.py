#!/usr/bin/env python3

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-coder")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(read_text(path))


def strip_markdown_fences(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def is_python_syntax_ok(src: str) -> Tuple[bool, str]:
    if not src or not src.strip():
        return False, "empty file"
    try:
        ast.parse(src)
        return True, ""
    except SyntaxError as e:
        return False, f"{e.msg} at line {e.lineno}, col {e.offset}"


def chat_with_deepseek(prompt: str, temperature: float = 0.1) -> str:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("Missing DEEPSEEK_API_KEY environment variable.")

    url = f"{DEEPSEEK_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert Django testing assistant. "
                    "You repair generated regression tests for real open-source projects."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "stream": False,
    }

    last_err: Optional[Exception] = None
    for attempt in range(1, 4):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=180)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:1000]}")
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e
            time.sleep(2 * attempt)

    raise RuntimeError(f"DeepSeek failed after retries: {last_err}")


def looks_like_django_test(code: str) -> Tuple[bool, str]:
    if "from solution import Solution" in code:
        return False, "LeetCode-style import found: from solution import Solution"

    if re.search(r"\bSolution\s*\(", code):
        return False, "LeetCode-style Solution() call found"

    has_unittest_class = bool(
        re.search(r"class\s+\w+\((?:.*TestCase|.*SimpleTestCase|.*TransactionTestCase).*\)\s*:", code)
    )
    has_test_method = bool(re.search(r"def\s+test_[A-Za-z0-9_]+\s*\(", code))

    if not has_test_method:
        return False, "No test_ method/function found"

    if not has_unittest_class:
        return False, "No Django/unittest-style TestCase class found"

    return True, ""


def fixer_agent(
    issue_text: str,
    scenarios: str,
    current_test: str,
    buggy_context: str,
    metadata: Dict[str, Any],
    last_error: str = "",
) -> str:
    prompt = f"""
You are repairing a generated regression test for a real Django SWE-bench task.

Goal:
Rewrite the current generated test into a runnable Django regression test file.

Rules:
- Output ONLY the complete Python test file.
- Do NOT use markdown fences.
- Do NOT generate or modify production code.
- Do NOT use fixed code or developer-written regression tests.
- Do NOT use LeetCode style.
- Do NOT import from solution.
- Do NOT call Solution().
- Use Django's unittest-style test format.
- Prefer SimpleTestCase if database is not needed.
- Prefer TestCase if database/models/querysets are needed.
- Each test method must start with test_.
- Keep the file minimal and runnable inside Django's tests/ directory.
- Avoid external dependencies.
- If defining temporary models, include app_label in Meta when needed.

Metadata:
{json.dumps(metadata, indent=2)}

Issue / functional requirement:
{issue_text}

Generated scenarios:
{scenarios}

Current generated test:
{current_test}

Previous validation or execution error:
{last_error}

Buggy-version context:
{buggy_context[:12000]}
""".strip()

    return strip_markdown_fences(chat_with_deepseek(prompt))


def copy_test_into_django(repo_path: Path, test_code: str, task_id: str) -> Path:
    test_dir = repo_path / "tests" / "testcopilot_generated"
    test_dir.mkdir(parents=True, exist_ok=True)

    init_file = test_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text("", encoding="utf-8")

    safe_task_id = re.sub(r"[^A-Za-z0-9_]+", "_", task_id)
    test_path = test_dir / f"test_generated_{safe_task_id}.py"
    write_text(test_path, test_code.strip() + "\n")
    return test_path


def run_django_test(repo_path: Path, env_name: str, module_name: str, timeout: int = 600) -> Dict[str, Any]:
    cmd = [
        "conda",
        "run",
        "-n",
        env_name,
        "python",
        "tests/runtests.py",
        module_name,
        "-v",
        "2",
    ]

    p = subprocess.run(
        cmd,
        cwd=str(repo_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )

    combined = (p.stdout or "") + "\n" + (p.stderr or "")

    if p.returncode == 0:
        status = "PASS_ON_BUGGY"
    elif re.search(r"\bFAIL\b|AssertionError|FAILED", combined, re.I):
        status = "FAIL_ON_BUGGY"
    else:
        status = "RUNTIME_ERROR"

    return {
        "cmd": " ".join(cmd),
        "returncode": p.returncode,
        "status": status,
        "stdout": p.stdout,
        "stderr": p.stderr,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--generated-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    task_dir = Path(args.task_dir).expanduser()
    generated_dir = Path(args.generated_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    issue_path = task_dir / "issue.md"
    metadata_path = task_dir / "metadata.json"
    generated_test_path = generated_dir / "generated_test.py"
    scenarios_path = generated_dir / "scenarios.md"
    buggy_context_path = generated_dir / "buggy_context.txt"

    if not issue_path.exists():
        raise FileNotFoundError(f"Missing issue.md: {issue_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.json: {metadata_path}")
    if not generated_test_path.exists():
        raise FileNotFoundError(f"Missing generated_test.py: {generated_test_path}")

    issue_text = read_text(issue_path)
    metadata = load_json(metadata_path)
    current_test = strip_markdown_fences(read_text(generated_test_path))
    scenarios = read_text(scenarios_path) if scenarios_path.exists() else ""
    buggy_context = read_text(buggy_context_path) if buggy_context_path.exists() else ""

    task_id = metadata.get("task_id", "unknown_task")
    repo_path = Path(metadata["repo_path"])
    env_name = metadata.get("env_name", "setup_django__django__3.0")

    if not repo_path.exists():
        raise RuntimeError(f"repo_path does not exist: {repo_path}")

    write_text(out_dir / "initial_generated_test.py", current_test)

    final_test = current_test
    validation_history: List[Dict[str, Any]] = []

    for attempt in range(0, args.max_retries + 1):
        print(f"[INFO] Validation attempt {attempt}")

        final_test = strip_markdown_fences(final_test)

        syntax_ok, syntax_err = is_python_syntax_ok(final_test)
        django_ok, django_err = looks_like_django_test(final_test) if syntax_ok else (False, syntax_err)

        local_status = {
            "attempt": attempt,
            "syntax_ok": syntax_ok,
            "syntax_error": syntax_err,
            "django_style_ok": django_ok,
            "django_style_error": django_err,
        }

        if syntax_ok and django_ok:
            test_path = copy_test_into_django(repo_path, final_test, task_id)
            print(f"[INFO] Copied test to: {test_path}")

            run_result = run_django_test(
                repo_path=repo_path,
                env_name=env_name,
                module_name="testcopilot_generated",
                timeout=args.timeout,
            )

            local_status["execution"] = run_result
            validation_history.append(local_status)

            write_text(out_dir / f"attempt_{attempt}_test.py", final_test)
            write_text(out_dir / f"attempt_{attempt}_stdout.txt", run_result.get("stdout", ""))
            write_text(out_dir / f"attempt_{attempt}_stderr.txt", run_result.get("stderr", ""))

            # Any executable result is useful. Stop on PASS_ON_BUGGY or FAIL_ON_BUGGY.
            if run_result["status"] in {"PASS_ON_BUGGY", "FAIL_ON_BUGGY"}:
                break

            last_error = run_result.get("stderr", "")[:4000]
        else:
            validation_history.append(local_status)
            last_error = f"syntax_ok={syntax_ok}, syntax_err={syntax_err}, django_ok={django_ok}, django_err={django_err}"

        if attempt >= args.max_retries:
            break

        print("[INFO] Calling fixer agent...")
        final_test = fixer_agent(
            issue_text=issue_text,
            scenarios=scenarios,
            current_test=final_test,
            buggy_context=buggy_context,
            metadata=metadata,
            last_error=last_error,
        )

    write_text(out_dir / "fixed_test_final.py", final_test)
    write_text(out_dir / "validation_history.json", json.dumps(validation_history, indent=2))

    final_status = "UNKNOWN"
    if validation_history:
        last = validation_history[-1]
        final_status = last.get("execution", {}).get("status", "LOCAL_INVALID")

    summary = {
        "task_id": task_id,
        "repo_path": str(repo_path),
        "env_name": env_name,
        "final_status": final_status,
        "final_test": str(out_dir / "fixed_test_final.py"),
        "history": str(out_dir / "validation_history.json"),
    }

    write_text(out_dir / "summary.json", json.dumps(summary, indent=2))

    print("\nDONE")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
