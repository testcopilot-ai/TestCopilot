#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


# =========================================================
# DeepSeek CONFIG
# =========================================================

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-coder")


# =========================================================
# DeepSeek Client
# =========================================================

class DeepSeekClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: int = 180,
        max_retries: int = 3,
        retry_sleep: float = 2.0,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

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

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(read_text(path))


def strip_markdown_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_section(text: str, header: str) -> str:
    pattern = rf"(?ims)^\s*{re.escape(header)}\s*:\s*$"
    m = re.search(pattern, text)
    if not m:
        return ""

    start = m.end()
    next_header = re.search(r"(?ims)^\s*(Scenarios|Testcases|Suggested location)\s*:\s*$", text[start:])
    if next_header:
        end = start + next_header.start()
    else:
        end = len(text)

    return text[start:end].strip()


def is_valid_python(code: str) -> Tuple[bool, str]:
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"{e.msg} at line {e.lineno}, col {e.offset}"


def tokenize_issue(issue_text: str) -> List[str]:
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", issue_text)
    stop = {
        "the", "and", "for", "with", "from", "this", "that", "when", "then",
        "should", "would", "could", "into", "error", "issue", "bug", "django",
        "test", "case", "value", "expected", "actual", "using", "while",
    }
    words = [w for w in words if w.lower() not in stop]
    # keep stable unique order
    out = []
    for w in words:
        if w not in out:
            out.append(w)
    return out[:50]


def score_file(path: Path, keywords: List[str]) -> int:
    try:
        text = read_text(path)
    except Exception:
        return 0

    score = 0
    lower = text.lower()
    name = str(path).lower()

    for kw in keywords:
        k = kw.lower()
        if k in name:
            score += 8
        score += lower.count(k)

    # Prefer source files, avoid tests because developer tests must not be used during generation.
    if "/tests/" in str(path):
        score -= 1000
    if "test_" in path.name:
        score -= 1000
    if path.name.startswith("test"):
        score -= 1000

    return score


def collect_buggy_context(repo_path: Path, issue_text: str, max_files: int = 8, max_chars_per_file: int = 5000) -> str:
    keywords = tokenize_issue(issue_text)

    py_files = []
    for p in repo_path.rglob("*.py"):
        sp = str(p)
        if "/.git/" in sp:
            continue
        if "/tests/" in sp:
            continue
        if "/docs/" in sp:
            continue
        if "/build/" in sp:
            continue
        py_files.append(p)

    ranked = sorted(py_files, key=lambda x: score_file(x, keywords), reverse=True)
    selected = [p for p in ranked if score_file(p, keywords) > 0][:max_files]

    if not selected:
        selected = ranked[:max_files]

    blocks = []
    for p in selected:
        try:
            rel = p.relative_to(repo_path)
        except Exception:
            rel = p

        content = read_text(p)[:max_chars_per_file]
        blocks.append(
            f"### File: {rel}\n"
            f"{content}\n"
        )

    return "\n\n".join(blocks)


def build_prompt(issue_text: str, metadata: Dict[str, Any], repo_context: str) -> List[Dict[str, str]]:
    system = """
You are an expert software testing researcher.
You generate regression tests for real open-source Python projects.
Return plain text only.
Do not use markdown fences.
Do not generate patches.
Do not modify production code.
""".strip()

    user = f"""
You are given a real SWE-bench/Django issue and buggy-version code context.

Task:
Generate scenario descriptions and one Django regression test file that can expose the issue.

Important rules:
- Use only the issue description and buggy-version context.
- Do NOT use fixed code.
- Do NOT use developer-written regression tests.
- Do NOT generate a patch.
- Generate a test that is intended to FAIL on the buggy version.
- Prefer Django's own test style: unittest-style assertions, SimpleTestCase/TestCase if needed.
- Keep the generated test self-contained as much as possible.
- If models are needed, define minimal local test models inside the test file only if Django supports that pattern.
- Return exactly these three sections:

Scenarios:
Testcases:
Suggested location:

Project metadata:
{json.dumps(metadata, indent=2)}

Issue / functional requirement:
{issue_text}

Buggy-version source context:
{repo_context}
""".strip()

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", required=True, help="Path to real_project_tasks/swebench_django_11133")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--max-files", type=int, default=8)
    parser.add_argument("--max-chars-per-file", type=int, default=5000)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    if not DEEPSEEK_API_KEY:
        raise SystemExit("ERROR: Please set DEEPSEEK_API_KEY environment variable.")

    task_dir = Path(args.task_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    issue_path = task_dir / "issue.md"
    metadata_path = task_dir / "metadata.json"

    if not issue_path.exists():
        raise FileNotFoundError(f"Missing issue file: {issue_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    issue_text = read_text(issue_path)
    metadata = load_json(metadata_path)

    repo_path_str = metadata.get("repo_path", "")
    if not repo_path_str:
        raise RuntimeError("metadata.json does not contain repo_path.")

    repo_path = Path(repo_path_str)
    if not repo_path.exists():
        raise RuntimeError(f"repo_path does not exist: {repo_path}")

    print(f"[INFO] Task dir: {task_dir}")
    print(f"[INFO] Repo path: {repo_path}")
    print("[INFO] Collecting buggy-version context...")

    repo_context = collect_buggy_context(
        repo_path=repo_path,
        issue_text=issue_text,
        max_files=args.max_files,
        max_chars_per_file=args.max_chars_per_file,
    )

    write_text(out_dir / "buggy_context.txt", repo_context)

    client = DeepSeekClient(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        model=DEEPSEEK_MODEL,
    )

    print("[INFO] Calling DeepSeek to generate scenarios and regression test...")
    raw = client.chat(
        build_prompt(issue_text=issue_text, metadata=metadata, repo_context=repo_context),
        temperature=args.temperature,
    )

    write_text(out_dir / "raw_model_output.txt", raw)

    scenarios = extract_section(raw, "Scenarios")
    testcases = extract_section(raw, "Testcases")
    suggested_location = extract_section(raw, "Suggested location")

    testcases = strip_markdown_fences(testcases)

    write_text(out_dir / "scenarios.md", scenarios)
    write_text(out_dir / "generated_test.py", testcases)
    write_text(out_dir / "suggested_location.txt", suggested_location)

    ok, err = is_valid_python(testcases)
    status = {
        "python_syntax_ok": ok,
        "syntax_error": err,
        "suggested_location": suggested_location,
        "output_test": str(out_dir / "generated_test.py"),
    }

    write_text(out_dir / "generation_status.json", json.dumps(status, indent=2))

    print("[DONE] Generated files:")
    print(f"  {out_dir / 'scenarios.md'}")
    print(f"  {out_dir / 'generated_test.py'}")
    print(f"  {out_dir / 'generation_status.json'}")

    if ok:
        print("[OK] generated_test.py is valid Python syntax.")
    else:
        print(f"[WARN] generated_test.py has syntax error: {err}")


if __name__ == "__main__":
    main()
