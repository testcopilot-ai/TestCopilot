#!/usr/bin/env python3

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

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        last_err: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=self.timeout)

                if r.status_code != 200:
                    raise RuntimeError(f"DeepSeek API error {r.status_code}: {r.text[:1200]}")

                return r.json()["choices"][0]["message"]["content"]

            except Exception as e:
                last_err = e
                print(f"[WARN] DeepSeek call failed attempt {attempt}/{self.max_retries}: {e}")
                time.sleep(self.retry_sleep * attempt)

        raise RuntimeError(f"DeepSeek request failed after retries: {last_err}")


# =========================================================
# File Helpers
# =========================================================

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(read_text(path))


def strip_markdown_fences(text: str) -> str:
    text = (text or "").strip()

    match = re.search(
        r"```(?:python|py)?\s*(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if match:
        return match.group(1).strip()

    text = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    return text.strip()


def extract_section(
    text: str,
    header: str,
    possible_headers: Optional[List[str]] = None,
) -> str:
    if possible_headers is None:
        possible_headers = [
            "Scenarios",
            "Testcases",
            "Suggested location",
            "Suggested Location",
            "Feedback",
            "Verdict",
            "Fixed Testcases",
        ]

    pattern = rf"(?ims)^\s*{re.escape(header)}\s*:\s*$"
    m = re.search(pattern, text or "")

    if not m:
        return ""

    start = m.end()

    next_header_pattern = "|".join(re.escape(h) for h in possible_headers)
    next_header = re.search(
        rf"(?ims)^\s*({next_header_pattern})\s*:\s*$",
        text[start:],
    )

    if next_header:
        end = start + next_header.start()
    else:
        end = len(text)

    return text[start:end].strip()


def is_valid_python(code: str) -> Tuple[bool, str]:
    if not code or not code.strip():
        return False, "empty generated test file"

    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"{e.msg} at line {e.lineno}, col {e.offset}"


def safe_json_loads(text: str) -> Dict[str, Any]:
    text = (text or "").strip()

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        text = match.group(0)

    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("JSON root is not an object")
        return data
    except Exception:
        return {
            "verdict": "FAIL",
            "feedback": "The Informant Agent did not return valid JSON.",
            "violations": ["invalid_informant_json"],
            "raw_output": text,
        }


def looks_like_patch_or_production_change(code: str) -> bool:
    markers = [
        "diff --git",
        "+++ ",
        "--- ",
        "@@ ",
        "git apply",
        "patch",
    ]
    low = code.lower()
    return any(m.lower() in low for m in markers)


def basic_test_style_check(code: str) -> Tuple[bool, str]:
    if "from solution import Solution" in code:
        return False, "LeetCode-style import found: from solution import Solution"

    if re.search(r"\bSolution\s*\(", code):
        return False, "LeetCode-style Solution() call found"

    if looks_like_patch_or_production_change(code):
        return False, "Output appears to contain a patch or production-code change"

    if not re.search(r"\bdef\s+test_[A-Za-z0-9_]+\s*\(", code):
        return False, "No test_ function or test_ method found"

    if not re.search(r"\bassert\b|self\.assert|assertRaises|with\s+self\.assertRaises", code):
        return False, "No meaningful assertion found"

    return True, ""


# =========================================================
# Context Collection
# =========================================================

def tokenize_issue(issue_text: str) -> List[str]:
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", issue_text or "")

    stop = {
        "the", "and", "for", "with", "from", "this", "that", "when", "then",
        "should", "would", "could", "into", "error", "issue", "bug", "django",
        "test", "case", "value", "expected", "actual", "using", "while",
        "project", "functional", "requirement", "reported", "behavior",
    }

    words = [w for w in words if w.lower() not in stop]

    out = []
    for w in words:
        if w not in out:
            out.append(w)

    return out[:60]


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

    # Avoid developer tests during generation.
    sp = str(path)
    if "/tests/" in sp:
        score -= 1000
    if "test_" in path.name:
        score -= 1000
    if path.name.startswith("test"):
        score -= 1000

    return score


def collect_buggy_context(
    repo_path: Path,
    issue_text: str,
    max_files: int = 8,
    max_chars_per_file: int = 5000,
) -> str:
    keywords = tokenize_issue(issue_text)

    py_files: List[Path] = []

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
        if "/dist/" in sp:
            continue
        if "/site-packages/" in sp:
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


# =========================================================
# Prompt 1: Scenario Generation Only
# =========================================================

def build_scenario_prompt(
    issue_text: str,
    metadata: Dict[str, Any],
    repo_context: str,
) -> List[Dict[str, str]]:
    system = """
You are an expert software testing researcher.
You generate scenario descriptions for real open-source Python projects.
Return plain text only.
Do not use markdown fences.
Do not generate executable tests in this step.
Do not generate patches.
Do not modify production code.
""".strip()

    user = f"""
You are given a real SWE-bench/Django issue and buggy-version source context.

Task:
Generate requirement-aligned test scenarios only.

Important rules:
- Use only the issue description and buggy-version context.
- Do NOT use fixed code.
- Do NOT use developer-written regression tests.
- Do NOT generate executable test code in this step.
- Do NOT generate a patch.
- Generate 5 to 8 scenarios.
- Each scenario must describe:
  1. the condition being tested,
  2. the expected behavior,
  3. why the scenario exposes the reported issue.
- Keep scenarios concrete enough that executable tests can be generated from them.

Return exactly this section:

Scenarios:

Project metadata:
{json.dumps(metadata, indent=2)}

Issue / functional requirement:
{issue_text}

Buggy-version source context:
{repo_context}
""".strip()

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# =========================================================
# Prompt 2: Scenario-Based Test Generation
# =========================================================

def build_test_generation_prompt(
    issue_text: str,
    metadata: Dict[str, Any],
    repo_context: str,
    scenarios: str,
) -> List[Dict[str, str]]:
    system = """
You are an expert Django regression-test generator.
You generate executable Django/Python regression tests from scenarios.
Return plain text only.
Do not generate patches.
Do not modify production code.
""".strip()

    user = f"""
You are given a real SWE-bench/Django issue, buggy-version source context, and scenario descriptions.

Task:
Generate one Django regression test file from the provided scenarios.

Important rules:
- Use only the issue description, buggy-version context, and generated scenarios.
- Do NOT use fixed code.
- Do NOT use developer-written regression tests.
- Do NOT generate a patch.
- The test should be intended to FAIL on the buggy version if the issue is present.
- Prefer Django's own test style: unittest-style assertions, SimpleTestCase/TestCase if needed.
- Keep the generated test self-contained as much as possible.
- If models are needed, define minimal local test models inside the test file only if Django supports that pattern.
- Do NOT use LeetCode style.
- Do NOT import from solution.
- Do NOT call Solution().
- Output executable Python test code only in the Testcases section.
- Also suggest the most likely test location.

Return exactly these two sections:

Testcases:
Suggested location:

Project metadata:
{json.dumps(metadata, indent=2)}

Issue / functional requirement:
{issue_text}

Generated scenarios:
{scenarios}

Buggy-version source context:
{repo_context}
""".strip()

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# =========================================================
# Prompt 3: Informant Agent
# =========================================================

def build_informant_prompt(
    issue_text: str,
    metadata: Dict[str, Any],
    scenarios: str,
    testcases: str,
    syntax_ok: bool,
    syntax_error: str,
    style_ok: bool,
    style_error: str,
) -> List[Dict[str, str]]:
    system = """
You are the Informant Agent for regression-test validation.
Your role is to check whether generated tests align with the issue requirement and testing methodology.
You do not repair tests.
Return JSON only.
""".strip()

    user = f"""
Check the generated Django regression test.

Validation criteria:
1. The test must align with the issue / functional requirement.
2. The test must be derived from the generated scenarios.
3. The test must not require fixed code.
4. The test must not contain a production-code patch.
5. The test must contain meaningful assertions.
6. The test must be consistent with Django regression-test methodology.
7. The test must be executable Python syntax.
8. The test must not use LeetCode style, from solution import Solution, or Solution().

Return JSON only in this format:
{{
  "verdict": "PASS" or "FAIL",
  "feedback": "short specific feedback",
  "violations": ["list of violated criteria, empty if PASS"]
}}

Project metadata:
{json.dumps(metadata, indent=2)}

Issue / functional requirement:
{issue_text}

Generated scenarios:
{scenarios}

Local checks:
syntax_ok = {syntax_ok}
syntax_error = {syntax_error}
style_ok = {style_ok}
style_error = {style_error}

Generated testcases:
{testcases}
""".strip()

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# =========================================================
# Prompt 4: Fixer Agent
# =========================================================

def build_fixer_prompt(
    issue_text: str,
    metadata: Dict[str, Any],
    repo_context: str,
    scenarios: str,
    testcases: str,
    informant_feedback: Dict[str, Any],
    syntax_error: str,
    style_error: str,
) -> List[Dict[str, str]]:
    system = """
You are the Fixer Agent for generated Django regression tests.
Your role is to repair the test file using Informant Agent feedback.
Return plain text only.
Do not generate patches.
Do not modify production code.
""".strip()

    user = f"""
Repair the generated Django regression test file.

Important rules:
- Use only the issue description, buggy-version context, scenarios, and Informant Agent feedback.
- Do NOT use fixed code.
- Do NOT use developer-written regression tests.
- Do NOT generate a patch.
- Do NOT change the issue requirement.
- Preserve the intended behavior from the scenarios.
- Fix alignment, assertion, methodology, import, or syntax problems.
- Return only executable Python code.
- Do not include explanations outside the code.
- Do NOT use LeetCode style.
- Do NOT import from solution.
- Do NOT call Solution().

Project metadata:
{json.dumps(metadata, indent=2)}

Issue / functional requirement:
{issue_text}

Generated scenarios:
{scenarios}

Buggy-version source context:
{repo_context}

Informant Agent feedback:
{json.dumps(informant_feedback, indent=2)}

Python syntax error, if any:
{syntax_error}

Local style error, if any:
{style_error}

Current generated testcases:
{testcases}
""".strip()

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# =========================================================
# Main Pipeline
# =========================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task-dir",
        required=True,
        help="Path to real_project_tasks/swebench_django_xxxxx",
    )

    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory",
    )

    parser.add_argument("--max-files", type=int, default=8)
    parser.add_argument("--max-chars-per-file", type=int, default=5000)
    parser.add_argument("--temperature-scenarios", type=float, default=0.2)
    parser.add_argument("--temperature-tests", type=float, default=0.2)
    parser.add_argument("--temperature-agents", type=float, default=0.0)
    parser.add_argument("--max-repair-rounds", type=int, default=3)

    args = parser.parse_args()

    if not DEEPSEEK_API_KEY:
        raise SystemExit("ERROR: Please set DEEPSEEK_API_KEY environment variable.")

    task_dir = Path(args.task_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
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

    repo_path = Path(repo_path_str).expanduser()

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

    # ---------------------------------------------------------
    # Step 1: Generate scenarios only
    # ---------------------------------------------------------

    print("[STEP 1] Generating scenarios...")

    raw_scenario_output = client.chat(
        build_scenario_prompt(
            issue_text=issue_text,
            metadata=metadata,
            repo_context=repo_context,
        ),
        temperature=args.temperature_scenarios,
    )

    write_text(out_dir / "raw_scenario_output.txt", raw_scenario_output)

    scenarios = extract_section(raw_scenario_output, "Scenarios")

    if not scenarios:
        scenarios = raw_scenario_output.strip()

    write_text(out_dir / "scenarios.md", scenarios)

    # ---------------------------------------------------------
    # Step 2: Generate scenario-based testcases
    # ---------------------------------------------------------

    print("[STEP 2] Generating scenario-based testcases...")

    raw_test_output = client.chat(
        build_test_generation_prompt(
            issue_text=issue_text,
            metadata=metadata,
            repo_context=repo_context,
            scenarios=scenarios,
        ),
        temperature=args.temperature_tests,
    )

    write_text(out_dir / "raw_test_generation_output.txt", raw_test_output)

    testcases = extract_section(raw_test_output, "Testcases")
    suggested_location = extract_section(raw_test_output, "Suggested location")

    if not testcases:
        testcases = raw_test_output.strip()

    testcases = strip_markdown_fences(testcases)

    write_text(out_dir / "generated_test_initial.py", testcases)
    write_text(out_dir / "suggested_location.txt", suggested_location)

    # ---------------------------------------------------------
    # Step 3 and 4: Informant + Fixer loop
    # ---------------------------------------------------------

    final_testcases = testcases
    final_informant_feedback: Dict[str, Any] = {}
    repair_history: List[Dict[str, Any]] = []

    for round_id in range(0, args.max_repair_rounds + 1):
        print(f"[STEP 3] Informant validation round {round_id}...")

        syntax_ok, syntax_error = is_valid_python(final_testcases)
        style_ok, style_error = basic_test_style_check(final_testcases)

        raw_informant_output = client.chat(
            build_informant_prompt(
                issue_text=issue_text,
                metadata=metadata,
                scenarios=scenarios,
                testcases=final_testcases,
                syntax_ok=syntax_ok,
                syntax_error=syntax_error,
                style_ok=style_ok,
                style_error=style_error,
            ),
            temperature=args.temperature_agents,
        )

        write_text(out_dir / f"informant_round_{round_id}.txt", raw_informant_output)

        informant_feedback = safe_json_loads(raw_informant_output)
        final_informant_feedback = informant_feedback

        verdict = str(informant_feedback.get("verdict", "FAIL")).strip().upper()

        repair_history.append(
            {
                "round": round_id,
                "syntax_ok": syntax_ok,
                "syntax_error": syntax_error,
                "style_ok": style_ok,
                "style_error": style_error,
                "informant_feedback": informant_feedback,
            }
        )

        if verdict == "PASS" and syntax_ok and style_ok:
            print("[OK] Informant passed the generated test.")
            break

        if round_id >= args.max_repair_rounds:
            print("[WARN] Maximum repair rounds reached.")
            break

        print("[STEP 4] Fixer repairing generated testcases...")

        raw_fixer_output = client.chat(
            build_fixer_prompt(
                issue_text=issue_text,
                metadata=metadata,
                repo_context=repo_context,
                scenarios=scenarios,
                testcases=final_testcases,
                informant_feedback=informant_feedback,
                syntax_error=syntax_error,
                style_error=style_error,
            ),
            temperature=args.temperature_agents,
        )

        write_text(out_dir / f"fixer_round_{round_id}.txt", raw_fixer_output)

        fixed_testcases = strip_markdown_fences(raw_fixer_output)

        if fixed_testcases.strip():
            final_testcases = fixed_testcases

        write_text(out_dir / f"generated_test_after_repair_round_{round_id}.py", final_testcases)

    # ---------------------------------------------------------
    # Save final outputs
    # ---------------------------------------------------------

    syntax_ok, syntax_error = is_valid_python(final_testcases)
    style_ok, style_error = basic_test_style_check(final_testcases)

    write_text(out_dir / "generated_test.py", final_testcases)

    final_status = {
        "python_syntax_ok": syntax_ok,
        "syntax_error": syntax_error,
        "style_ok": style_ok,
        "style_error": style_error,
        "suggested_location": suggested_location,
        "informant_final_feedback": final_informant_feedback,
        "repair_history": repair_history,
        "output_test": str(out_dir / "generated_test.py"),
        "output_scenarios": str(out_dir / "scenarios.md"),
    }

    write_text(out_dir / "generation_status.json", json.dumps(final_status, indent=2))

    print("[DONE] Generated files:")
    print(f"  {out_dir / 'scenarios.md'}")
    print(f"  {out_dir / 'generated_test_initial.py'}")
    print(f"  {out_dir / 'generated_test.py'}")
    print(f"  {out_dir / 'generation_status.json'}")

    if syntax_ok:
        print("[OK] final generated_test.py is valid Python syntax.")
    else:
        print(f"[WARN] final generated_test.py has syntax error: {syntax_error}")

    if style_ok:
        print("[OK] final generated_test.py passed local style checks.")
    else:
        print(f"[WARN] final generated_test.py failed style check: {style_error}")

    final_verdict = str(final_informant_feedback.get("verdict", "UNKNOWN")).upper()
    print(f"[INFO] Final Informant verdict: {final_verdict}")


if __name__ == "__main__":
    main()
