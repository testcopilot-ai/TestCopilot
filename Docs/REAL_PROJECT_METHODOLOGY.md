# Methodology Document: Real-Project Validation of TestCopilot on SWE-bench/Django

## 1. Purpose

This document explains how the real-project Django/SWE-bench validation is used inside the TestCopilot methodology. The goal is to address the reviewer concern that the original evaluation mainly used natural-language benchmark tasks whose solutions are usually single-file or single-function. This real-project validation demonstrates that TestCopilot can also generate regression tests for a mature open-source project with multiple files, framework dependencies, and project-level behavior.

The selected real-project case is:

| Field | Value |
|---|---|
| Task ID | `django__django-11133` |
| Project | Django |
| Source | SWE-bench task instance |
| Behavior under test | `HttpResponse` should handle `memoryview` content as bytes |
| Buggy result | FAIL |
| Fixed result | PASS |
| Classification | Bug-revealing regression test |

## 2. High-Level Methodology

The Django/SWE-bench validation follows the same core TestCopilot idea: generate test scenarios first, use the scenarios to guide test-case generation, and validate the generated tests with an agent-based quality loop.

The workflow is:

```text
SWE-bench issue description + buggy Django repository context
        ↓
Scenario Generator
        ↓
Requirement-aligned scenarios
        ↓
Scenario-based Test Generator
        ↓
Initial Django regression test file
        ↓
Informant Agent
        ↓
PASS? ── yes ──→ final generated test
  │
  no
  ↓
Fixer Agent receives Informant feedback
        ↓
Repaired generated test
        ↓
Buggy-version execution
        ↓
Fixed-version execution
```

AutoCodeRover/SWE-bench infrastructure is used only to obtain a realistic executable Django task environment. TestCopilot is not used to generate patches. It uses the issue description and buggy-version context to generate regression tests.

## 3. Inputs Used by TestCopilot

For each real-project task, TestCopilot receives only information available before the fix:

1. SWE-bench issue description.
2. Project metadata such as task ID, project name, version, and base commit.
3. Buggy-version source context extracted from the repository.
4. Generated scenario descriptions.
5. Generated tests from the scenario-based test generator.
6. Informant feedback during repair.

The following information is intentionally excluded during generation:

1. Fixed source code.
2. Official patch content.
3. Developer-written regression tests.
4. Any fail-to-pass oracle from SWE-bench.

This prevents leakage and keeps the evaluation consistent with the paper methodology.

## 4. Repository Files

Recommended repository structure:

```text
TestCopilot/
├── real_project_tasks/
│   └── swebench_django_11133/
│       ├── issue.md
│       └── metadata.json
├── scripts/
│   ├── generate_swebench_scenarios_agents.py
│   ├── fix_validate_swebench_django_tests.py
│   └── validate_swebench_fixed_version.py
├── results/
│   └── django_11133/
│       ├── scenarios.md
│       ├── generated_test_initial.py
│       ├── generated_test.py
│       ├── generation_status.json
│       ├── validation_history.json
│       ├── buggy_validation_output.txt
│       └── fixed_validation_output.txt
└── docs/
    └── REAL_PROJECT_METHODOLOGY.md
```

## 5. Task Preparation

The SWE-bench setup prepares the executable Django environment.

```bash
cd ~/ScenarioGenerated/SWE-bench
conda activate swe-bench

echo django__django-11133 > tasks.txt

python harness/run_setup.py \
  --log_dir logs \
  --testbed testbed \
  --result_dir setup_result \
  --subset_file tasks.txt
```

The expected setup outputs are:

```text
setup_result/setup_map.json
setup_result/tasks_map.json
testbed/django__django/setup_django__django__3.0/
```

If the Django clone is incomplete, manually clone the Django repository and checkout the SWE-bench base commit:

```bash
cd ~/ScenarioGenerated/SWE-bench/testbed/django__django
rm -rf setup_django__django__3.0

git clone https://gh-proxy.com/https://github.com/django/django.git setup_django__django__3.0
cd setup_django__django__3.0

git checkout 879cc3da6249e920b8d54518a0ae06de835d7373

conda run -n setup_django__django__3.0 python -m pip install -e .
```

Verify the repo is complete:

```bash
find ~/ScenarioGenerated/SWE-bench/testbed/django__django/setup_django__django__3.0 \
  -maxdepth 3 -name "runtests.py"
```

Expected:

```text
./tests/runtests.py
```

## 6. Scenario Generation and Scenario-Based Test Generation

The main generation script is:

```text
generate_swebench_scenarios_agents.py
```

It performs four internal stages:

1. Scenario generation only.
2. Test-case generation from the scenarios.
3. Informant Agent validation.
4. Fixer Agent repair if the Informant returns FAIL.

Run:

```bash
cd ~/ScenarioGenerated
conda activate furqan_env

export DEEPSEEK_API_KEY="YOUR_KEY_HERE"
export DEEPSEEK_MODEL="deepseek-coder"

python generate_swebench_scenarios_agents.py \
  --task-dir ~/ScenarioGenerated/real_project_tasks/swebench_django_11133 \
  --out-dir ~/ScenarioGenerated/testcopilot_real_project_output/django_11133 \
  --max-files 8 \
  --max-chars-per-file 5000 \
  --max-repair-rounds 3
```

Expected generated artifacts:

```text
scenarios.md
generated_test_initial.py
generated_test.py
generation_status.json
raw_scenario_output.txt
raw_test_generation_output.txt
informant_round_0.txt
fixer_round_0.txt
```

## 7. Informant Agent Role

The Informant Agent does not repair code. It checks whether the generated test satisfies the methodology:

1. The test aligns with the issue requirement.
2. The test is derived from generated scenarios.
3. The test does not require fixed code.
4. The test does not contain a production patch.
5. The test contains meaningful assertions.
6. The test follows Django regression-test style.
7. The test is executable Python.
8. The test does not use LeetCode-style `from solution import Solution` or `Solution()` calls.

If the Informant returns PASS and the local syntax/style checks also pass, the generated test proceeds to execution. If not, the test and feedback are passed to the Fixer Agent.

## 8. Fixer Agent Role

The Fixer Agent repairs the generated test using:

1. The original issue description.
2. Buggy-version context.
3. Generated scenarios.
4. Informant feedback.
5. Local syntax/style errors.

The Fixer Agent is not allowed to use fixed code, developer tests, or generate production patches. Its only output is a repaired executable Python regression test.

## 9. Buggy-Version Validation

After generation, the final test is copied into the Django testbed and executed.

```bash
cd ~/ScenarioGenerated
conda activate furqan_env

python fix_validate_swebench_django_tests.py \
  --task-dir ~/ScenarioGenerated/real_project_tasks/swebench_django_11133 \
  --generated-dir ~/ScenarioGenerated/testcopilot_real_project_output/django_11133 \
  --out-dir ~/ScenarioGenerated/testcopilot_real_project_output/django_11133_validation \
  --max-retries 2
```

The buggy result is valid only if the test fails due to assertion failures related to the reported behavior, not because of environment errors. For this task, the generated tests failed on the buggy version because `HttpResponse(memoryview(...))` returned a memoryview string representation instead of the underlying bytes.

Observed buggy behavior:

```text
b'<memory at ...>' != b'My Content'
b'<memory at ...>' != b'Test Data'
b'<memory at ...>' != b'Updated'
```

Therefore:

```text
Buggy version: FAIL
```

## 10. Fixed-Version Validation

After applying the official fix, the same generated test suite is executed again.

```bash
cd /home/furqan/ScenarioGenerated/testcopilot_real_project_output/django_11133_fixed_repo

conda run -n setup_django__django__3.0 \
  python tests/runtests.py testcopilot_generated -v 2
```

Observed fixed result:

```text
test_memoryview_content_multiple_access ... ok
test_memoryview_content_returns_bytes ... ok
test_memoryview_content_setter ... ok
test_memoryview_empty_bytes ... ok
test_memoryview_iteration ... ok

Ran 5 tests in 0.001s
OK
```

Therefore:

```text
Fixed version: PASS
```

## 11. Final Result

| Task | Project | Generated Tests | Buggy Version | Fixed Version | Classification |
|---|---:|---:|---:|---:|---|
| `django__django-11133` | Django | 5 | FAIL | PASS | Bug-revealing regression test |

This result directly supports the claim that TestCopilot can generate useful regression tests for a real multi-file open-source project.



## 12. Security and Reproducibility Notes

1. Do not commit API keys to the repository.
2. Use environment variables such as `DEEPSEEK_API_KEY`.
3. Rotate any key that was previously pasted in a terminal or shared in logs.
4. Store raw outputs and validation logs in `results/`, but remove secrets before publishing.
5. Keep fixed-code patches separate from generation inputs to avoid leakage.

Example `.env.example`:

```bash
DEEPSEEK_API_KEY=your_key_here
DEEPSEEK_MODEL=deepseek-coder
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
```
