# Real-Project Validation of TestCopilot on Django/SWE-bench

## 1. Purpose

This document describes the real-project validation of **TestCopilot** using Django/SWE-bench tasks.

The purpose of this validation is to address the reviewer concern that the original evaluation mainly used benchmark tasks where the target program is often a single file or a single function. To strengthen the evaluation, we additionally validate TestCopilot on real Django tasks from SWE-bench and real Django ticket-based validation.

Django is a mature open-source framework with a large multi-file codebase, project-level dependencies, framework-specific test infrastructure, and realistic regression-test requirements. Therefore, passing this validation shows that TestCopilot is not limited to isolated programming tasks but can also generate useful regression tests for real software projects.

---

## 2. Validated Real-Project Tasks

The current real-project validation includes three Django tasks.

| Task ID                | Project | Behavior Under Test                                                                                                            | Generated Tests | Buggy Version | Fixed Version | Result                        |
| ---------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------ | --------------: | ------------- | ------------- | ----------------------------- |
| `django__django-11133` | Django  | `HttpResponse` should handle `memoryview` content as bytes                                                                     |               5 | FAIL          | PASS          | Bug-revealing regression test |
| `django_32332`         | Django  | `ForeignKey` should correctly synchronize a related object whose non-numeric primary key is assigned after relation assignment |               6 | FAIL          | PASS          | Bug-revealing regression test |
| `django_32347`         | Django  | `ModelChoiceField` invalid-choice errors should include the submitted value in the validation error parameters/message         |               6 | FAIL          | PASS          | Bug-revealing regression test |

A generated test is counted as successful only if it satisfies the following condition:

```text
Buggy checkout: FAIL
Fixed checkout: PASS
```

This means the generated test exposes the original bug and remains valid after the developer fix.

---

## 3. Repository Structure

The real-project validation files are organized under:

```text
Real_project_validation/
```

Current repository structure:

```text
Real_project_validation/
├── Docs/
│   └── README_REAL_PROJECT_VALIDATION.md
├── django_11133/
│   ├── generated_test.py
│   ├── generated_test_initial.py
│   ├── scenarios.md
│   ├── generation_status.json
│   ├── raw_scenario_output.txt
│   ├── raw_test_generation_output.txt
│   ├── informant_round_0.txt
│   └── fixer_round_0.txt
├── django_11133_validation/
│   ├── summary.json
│   ├── validation_history.json
│   ├── fixed_test_final.py
│   ├── attempt_0_stdout.txt
│   └── attempt_0_stderr.txt
├── real_project_tasks/
│   └── swebench_django_11133/
│       ├── issue.md
│       └── metadata.json
├── scripts/
│   ├── generate_swebench_scenarios_agents.py
│   ├── fix_validate_swebench_django_tests.py
│   ├── validate_swebench_fixed_version.py
│   └── validate_manual_django_ticket_fixed_checkout.py
├── django_11133_gold_patch.diff
└── django_11133_real_project_summary.md
```

For the additional validated Django tasks, the following folders should also be included or reported in the same validation structure:

```text
Real_project_validation/
├── django_32332/
│   └── generated_test.py
├── django_32332_validation/
│   ├── summary.json
│   └── validation_history.json
├── django_32332_fixed_validation/
│   ├── fixed_summary.json
│   ├── fixed_stdout.txt
│   └── fixed_stderr.txt
├── django_32347/
│   └── generated_test.py
├── django_32347_validation/
│   ├── summary.json
│   └── validation_history.json
└── django_32347_fixed_validation/
    ├── fixed_summary.json
    ├── fixed_stdout.txt
    └── fixed_stderr.txt
```

---

## 4. High-Level Methodology

The real-project validation follows the same TestCopilot workflow used in the main evaluation.

```text
Issue description + buggy repository context
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

The key point is that TestCopilot generates **tests**, not patches. The fixed version is used only after test generation to validate whether the generated regression test passes after the official developer fix.

---

## 5. Inputs Used by TestCopilot

For each real-project task, TestCopilot uses only information available before the fix:

1. Issue description.
2. Project metadata.
3. Buggy repository context.
4. Scenario descriptions generated by the Scenario Generator.
5. Initial tests generated by the Scenario-Based Test Generator.
6. Informant Agent feedback.
7. Syntax, style, and execution feedback from the local validation loop.

The following information is intentionally excluded during generation:

1. Fixed source code.
2. Official patch content.
3. Developer-written regression tests.
4. SWE-bench fail-to-pass oracle.
5. Any test written after inspecting the fixed version.

This prevents leakage and keeps the validation consistent with the TestCopilot methodology.

---

## 6. Agent Roles

### 6.1 Scenario Generator

The Scenario Generator converts the issue description and buggy source context into requirement-aligned testing scenarios. These scenarios describe expected behavior and important edge cases.

### 6.2 Scenario-Based Test Generator

The Scenario-Based Test Generator converts the generated scenarios into executable Django regression tests.

### 6.3 Informant Agent

The Informant Agent checks whether the generated test is suitable for real-project validation.

It checks that:

1. The test aligns with the issue requirement.
2. The test is derived from generated scenarios.
3. The test does not require fixed code.
4. The test does not contain a production patch.
5. The test includes meaningful assertions.
6. The test follows Django regression-test style.
7. The test is executable Python.
8. The test does not use LeetCode-style imports such as:

```python
from solution import Solution
```

### 6.4 Fixer Agent

If the Informant Agent or the local validation loop finds problems, the Fixer Agent repairs the test. The Fixer Agent receives the issue description, buggy source context, generated scenarios, Informant feedback, and local execution errors.

The Fixer Agent is only allowed to repair the test file. It is not allowed to generate or modify production code.

---

## 7. Task 1: `django__django-11133`

### 7.1 Task Description

| Field               | Value                                                      |
| ------------------- | ---------------------------------------------------------- |
| Task ID             | `django__django-11133`                                     |
| Project             | Django                                                     |
| Source              | SWE-bench task instance                                    |
| Behavior under test | `HttpResponse` should handle `memoryview` content as bytes |
| Generated tests     | 5                                                          |
| Buggy result        | FAIL                                                       |
| Fixed result        | PASS                                                       |
| Classification      | Bug-revealing regression test                              |

### 7.2 Generation Command

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

### 7.3 Generated Artifacts

```text
django_11133/
├── scenarios.md
├── generated_test_initial.py
├── generated_test.py
├── generation_status.json
├── raw_scenario_output.txt
├── raw_test_generation_output.txt
├── informant_round_0.txt
└── fixer_round_0.txt
```

### 7.4 Buggy-Version Validation

```bash
cd ~/ScenarioGenerated
conda activate furqan_env

python fix_validate_swebench_django_tests.py \
  --task-dir ~/ScenarioGenerated/real_project_tasks/swebench_django_11133 \
  --generated-dir ~/ScenarioGenerated/testcopilot_real_project_output/django_11133 \
  --out-dir ~/ScenarioGenerated/testcopilot_real_project_output/django_11133_validation \
  --max-retries 2
```

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

### 7.5 Fixed-Version Validation

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

---

## 8. Task 2: `django_32332`

### 8.1 Task Description

| Field               | Value                                                                                                                          |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Task ID             | `django_32332`                                                                                                                 |
| Project             | Django                                                                                                                         |
| Behavior under test | `ForeignKey` should correctly synchronize a related object whose non-numeric primary key is assigned after relation assignment |
| Generated tests     | 6                                                                                                                              |
| Buggy result        | FAIL                                                                                                                           |
| Fixed result        | PASS                                                                                                                           |
| Classification      | Bug-revealing regression test                                                                                                  |

### 8.2 Bug-Revealing Behavior

The generated test checks that a child model using a `ForeignKey` correctly tracks the primary key of a related parent object when the parent’s character primary key is assigned after the relation has already been set.

The generated test covers multiple character primary-key cases:

```text
alpha
sku-001
SkuMixedCase
A100B200
product_key_05
ticket32332
```

### 8.3 Buggy-Version Validation

Before buggy validation, reinstall the buggy repository into the task environment:

```bash
conda run -n setup_django_32332 \
  python -m pip install -e ~/ScenarioGenerated/real_project_repos/django_32332
```

Then run:

```bash
python fix_validate_swebench_django_tests.py \
  --task-dir ~/ScenarioGenerated/real_project_tasks/django_32332 \
  --generated-dir ~/ScenarioGenerated/testcopilot_real_project_output/django_32332 \
  --out-dir ~/ScenarioGenerated/testcopilot_real_project_output/django_32332_validation \
  --max-retries 0
```

Observed result:

```text
final_status: FAIL_ON_BUGGY
```

### 8.4 Fixed-Version Validation

```bash
python validate_manual_django_ticket_fixed_checkout.py \
  --task-dir ~/ScenarioGenerated/real_project_tasks/django_32332 \
  --validation-dir ~/ScenarioGenerated/testcopilot_real_project_output/django_32332_validation \
  --out-dir ~/ScenarioGenerated/testcopilot_real_project_output/django_32332_fixed_validation
```

Observed fixed output:

```text
test_char_pk_alpha ... ok
test_char_pk_alphanumeric ... ok
test_char_pk_hyphen ... ok
test_char_pk_long_value ... ok
test_char_pk_mixed_case ... ok
test_char_pk_underscore ... ok

Ran 6 tests in 0.033s

OK
```

Observed result:

```text
status: PASS_ON_FIXED
```

---

## 9. Task 3: `django_32347`

### 9.1 Task Description

| Field               | Value                                                                                                                  |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Task ID             | `django_32347`                                                                                                         |
| Project             | Django                                                                                                                 |
| Behavior under test | `ModelChoiceField` invalid-choice errors should include the submitted value in the validation error parameters/message |
| Generated tests     | 6                                                                                                                      |
| Buggy result        | FAIL                                                                                                                   |
| Fixed result        | PASS                                                                                                                   |
| Classification      | Bug-revealing regression test                                                                                          |

### 9.2 Bug-Revealing Behavior

The generated test checks that invalid submitted values to `ModelChoiceField` are correctly included in the validation error message.

The generated test covers multiple invalid values:

```text
invalid
12345
missing-choice
MissingChoice
bad.choice
ticket32347-invalid-value
```

### 9.3 Buggy-Version Validation

Before buggy validation, reinstall the buggy repository into the task environment:

```bash
conda run -n setup_django_32347 \
  python -m pip install -e ~/ScenarioGenerated/real_project_repos/django_32347
```

Then run:

```bash
python fix_validate_swebench_django_tests.py \
  --task-dir ~/ScenarioGenerated/real_project_tasks/django_32347 \
  --generated-dir ~/ScenarioGenerated/testcopilot_real_project_output/django_32347 \
  --out-dir ~/ScenarioGenerated/testcopilot_real_project_output/django_32347_validation \
  --max-retries 0
```

Observed result:

```text
final_status: FAIL_ON_BUGGY
```

### 9.4 Fixed-Version Validation

```bash
python validate_manual_django_ticket_fixed_checkout.py \
  --task-dir ~/ScenarioGenerated/real_project_tasks/django_32347 \
  --validation-dir ~/ScenarioGenerated/testcopilot_real_project_output/django_32347_validation \
  --out-dir ~/ScenarioGenerated/testcopilot_real_project_output/django_32347_fixed_validation
```

Observed fixed output:

```text
test_invalid_choice_hyphenated_value ... ok
test_invalid_choice_long_value ... ok
test_invalid_choice_mixed_case_value ... ok
test_invalid_choice_number_like_string ... ok
test_invalid_choice_symbol_value ... ok
test_invalid_choice_text ... ok

Ran 6 tests in 0.019s

OK
```

Observed result:

```text
status: PASS_ON_FIXED
```

---

## 10. Final Real-Project Validation Results

| Task ID                | Generated Tests | Buggy Version | Fixed Version | Classification                |
| ---------------------- | --------------: | ------------- | ------------- | ----------------------------- |
| `django__django-11133` |               5 | FAIL          | PASS          | Bug-revealing regression test |
| `django_32332`         |               6 | FAIL          | PASS          | Bug-revealing regression test |
| `django_32347`         |               6 | FAIL          | PASS          | Bug-revealing regression test |

Final summary:

```text
django__django-11133
Generated tests: 5
Buggy version: FAIL
Fixed version: PASS

django_32332
Generated tests: 6
Buggy version: FAIL_ON_BUGGY
Fixed version: PASS_ON_FIXED

django_32347
Generated tests: 6
Buggy version: FAIL_ON_BUGGY
Fixed version: PASS_ON_FIXED
```

These results show that TestCopilot can generate regression tests that expose real bugs in a mature multi-file software project and pass after the corresponding developer fixes.

---

## 11. Validity Criteria

A real-project generated test is counted as valid only if:

1. It is executable by Django’s native test runner.
2. It contains meaningful assertions.
3. It fails on the buggy checkout.
4. It passes on the fixed checkout.
5. The buggy failure is caused by the reported behavior, not by syntax, import, or environment errors.
6. The same generated test file is used for both buggy and fixed validation.
7. The test does not include any production patch.
8. The test does not use fixed-code information during generation.

---

## 12. Environment Hygiene Note

During validation, we observed that fixed-version validation installs the fixed checkout into the conda environment in editable mode. If the same environment is later reused for buggy validation without reinstalling the buggy checkout, the buggy run may accidentally import Django from the fixed repository.

This can incorrectly produce:

```text
PASS_ON_BUGGY
```

To avoid this, always reinstall the buggy repository before buggy validation:

```bash
conda run -n setup_$TASK python -m pip install -e ~/ScenarioGenerated/real_project_repos/$TASK
```

Also verify the Django import path:

```bash
conda run -n setup_$TASK python - <<PY
import django
print(django.__file__)
PY
```

For buggy validation, the import path must point to:

```text
~/ScenarioGenerated/real_project_repos/<TASK>/django/__init__.py
```

---

## 13. Security and Reproducibility Notes

1. Do not commit API keys to the repository.
2. Use environment variables such as `DEEPSEEK_API_KEY`.
3. Rotate any key that was pasted in a terminal or shared in logs.
4. Store raw outputs and validation logs, but remove secrets before publishing.
5. Keep fixed-code patches separate from generation inputs to prevent leakage.

Example `.env.example`:

```bash
DEEPSEEK_API_KEY=your_key_here
DEEPSEEK_MODEL=deepseek-coder
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
```

---

