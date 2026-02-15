Perfect — you want the **same style and structure as your original version**, but polished, consistent, and academically clean — while keeping that GitHub-friendly format.

Below is your **complete refined README.md**, written in the same format and tone as your example.

You can copy–paste directly.

---

````md
# Replication Package for Paper  
# "Scenario-Driven Test Case Generation with Autonomous Agents"

![Architecture](TestCopilot.PNG)

# TestCopilot  
Scenario-enriched LLM-based framework for automatic test case generation, bug detection, and code evaluation.

---

# 🚀 Overview

**TestCopilot** is a scenario-driven, multi-agent framework designed to automate software test case generation using Large Language Models (LLMs).

Instead of directly translating requirements into tests, TestCopilot:

- Synthesizes structured execution scenarios
- Converts scenarios into executable pytest test cases
- Applies an autonomous Informant + Fixer validation loop
- Evaluates test effectiveness using coverage, mutation, and maintainability metrics

The framework is benchmarked on **HumanEval**, **MBPP**, and **LeetCode**, demonstrating strong performance across correctness, coverage, and robustness dimensions.

---

# 📊 Performance Highlights

```plaintext
TestCopilot significantly outperforms baseline models:

Metric                              TestCopilot
---------------------------------------------------------
TCE (HumanEval)                     99.3%
Coverage (HumanEval)                99.5%
Bugs Detected (HumanEval)           179
False Alarms (HumanEval)            0
Maintainability Index (HumanEval)   71.30%

TCE (MBPP)                          99.3%
Coverage (MBPP)                     99.5%
Bugs Detected (MBPP)                157
False Alarms (MBPP)                 0
Maintainability Index (MBPP)        83.31%

TCE (LeetCode)                      64.8%
Coverage (LeetCode)                 92.7%
Bugs Detected (LeetCode)            178
False Alarms (LeetCode)             43
Maintainability Index (LeetCode)    47.5%
````

> Total evaluation scale: **58,912 generated test cases**

---

# 🔑 Key Features

## ✅ Scenario-Enriched Prompting

Transforms abstract functional requirements into structured behavioral scenarios:

```
Input Conditions → Execution Context → Expected Outcome
```

This improves oracle fidelity and edge-case coverage.

---

## 🧠 Multi-Agent Evaluation

Includes specialized agents:

* **Generator Agent** – Produces initial candidate tests
* **Informant Agent** – Validates structure and semantic consistency
* **Fixer Agent** – Repairs invalid or incomplete tests
* **Evaluation Module** – Computes effectiveness and robustness metrics

Repair attempts are bounded (max retries = 3).

---

## 📈 Deep Coverage & Quality Analysis

TestCopilot computes:

* Test Case Effectiveness (TCE)
* False Alarm Rate (FAR)
* Function Coverage
* Statement Coverage
* Branch Coverage
* Path Coverage
* Maintainability Index (radon)

---

## 🔄 Test Repair Feedback Loop

When tests are invalid or inconsistent:

1. Informant identifies structural or semantic issues
2. Fixer minimally rewrites test cases
3. Validation is repeated (bounded retries)

Correct reference implementations are **never exposed** to repair agents.

---

# 📂 Repository Structure

```plaintext
TestCopilot/
│
├── 📂 dataset/                         # HumanEval / MBPP / LeetCode datasets
│   ├── HumanEval_Scenario_testcases.xlsx
│   ├── MBPP_Scenario_testcases.xlsx
│   ├── leetcode_dataset.csv
│
├── 📂 LLM-Based Evaluation/            # Multi-agent evaluation & robustness
│   ├── mainchatgpt.py                  # Evaluation with GPT-4 backend
│   ├── maindeepseek.py                 # Evaluation with DeepSeek backend
│   ├── mainleetcodeagentai.py          # LeetCode agent evaluation
│   ├── compute_repair_vs_discard.py    # Repair vs discard-fail comparison
│   ├── compute_temp_token.py           # Temperature/token variation analysis
│   ├── reasoningandnonreasoning.py     # Reasoning vs non-reasoning study
│   ├── semantic_fidelity.py            # Semantic fidelity evaluation
│   ├── stats_robustness.py             # Statistical robustness analysis
│
├── 📂 baseline/                        # Baseline evaluations & metrics
│   ├── main.py
│   ├── mainaibugy.py
│   ├── mainbugsapproach.py
│   ├── maincompute_pyuguinmetrics.py
│   ├── maincoveragezero.py
│   ├── mainmaintainabilty.py
│   ├── mainpyuguin.py
│   ├── mainpyuguin_mutation.py
│   ├── mainstatandfunccov.py
│
├── 📂 scenariogenerated/               # Scenario generation pipeline
│   ├── main.py
│   ├── mainscenarios.py
│   ├── mainscenariometrics.py
│
├── .env                                # API keys (OpenAI / DeepSeek)
├── requirements.txt                    # Dependencies
└── README.md                           # Documentation
```

---

# 📌 Requirements

## 🖥️ System Requirements

* Python 3.9+
* Internet access for LLM API calls

---

## 📦 Installation

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

# 🔑 API Configuration

Set your API key before running evaluation.

### Linux / macOS

```bash
export OPENAI_API_KEY=your_key_here
```

### Windows

```powershell
setx OPENAI_API_KEY "your_key_here"
```

---

# ▶️ Running TestCopilot

To run TestCopilot on your dataset:

```
Step #1 – Scenario Generation

python scenariogenerated/main.py \
  --input dataset/HumanEval_Scenario_testcases.xlsx \
  --output outputs/scenarios


Step #2 – Multi-Agent Evaluation (GPT-4 backend)

python "LLM-Based Evaluation/mainchatgpt.py" \
  --input outputs/scenarios \
  --output outputs/evaluated_tests
```

For DeepSeek backend:

```
python "LLM-Based Evaluation/maindeepseek.py" \
  --input outputs/scenarios \
  --output outputs/evaluated_tests
```

---

# 🔬 Reproducing Paper Analyses

### Repair vs Discard-Fail

```
python "LLM-Based Evaluation/compute_repair_vs_discard.py"
```

### Temperature / Token Sensitivity

```
python "LLM-Based Evaluation/compute_temp_token.py"
```

### Reasoning vs Non-Reasoning Study

```
python "LLM-Based Evaluation/reasoningandnonreasoning.py"
```

### Statistical Robustness

```
python "LLM-Based Evaluation/stats_robustness.py"
```

---







