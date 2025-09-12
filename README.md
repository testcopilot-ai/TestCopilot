# Replication package for paper "Scenario-Driven Test Case Generation withAutonomous Agents"

![Architecture](TestCopilot.png)
# TestCopilot
Scenario-enriched LLM-based framework for automatic test case generation, bug detection, and code evaluation.

Scenario-Driven Test Case Generation with Autonomous Agents

# 🚀 Overview

TestCopilot is a multi-agent framework designed to automate software test case generation using Large Language Models (LLMs). It integrates scenario-enriched prompting, bug detection, and coverage analysis to produce high-quality and maintainable test suites. The system is benchmarked on HumanEval and MBPP, demonstrating state-of-the-art performance across correctness, coverage, and maintainability metrics.


# 📊  Performance Highlights

```plaintext
TestCopilot significantly outperforms existing models:
Metric	TestCopilot
TCE (HumanEval)	99.3%
DDP (HumanEval)	99.7%
Function Coverage 89.5%
Bugs Detected (HumanEval) 179
Maintainability Index 81.31%
False Alarms 0
```
🔑 Key Features
# ✅ Scenario-Enriched Prompting

Integrates functional requirements and user stories to guide LLMs in generating purpose-driven test cases.
🧠 Multi-Agent Evaluation

Includes separate agents for generation, bug detection, test refinement, and repair validation.
📈 Deep Coverage Analysis

Calculates statement, branch, path, and integration coverage along with maintainability scores.
🔄 Test Repair Feedback Loop

Fixes incomplete or incorrect test cases using iterative improvement agents.
```plaintext
TestCopilot/
│
├── 📂 dataset/ # HumanEval / MBPP benchmark datasets
│ ├── HumanEval_Scenario_testcases.xlsx
│ ├── MBPP_Scenario_testcases.xlsx
│
├── 📂 LLM-Based Evaluation/ # Multi-agent evaluation & robustness
│ ├── compute_repair_vs_discard.py # Repair vs discard-fail comparison
│ ├── compute_temp_token.py # Temp/token variation analysis
│ ├── mainchatgpt.py # Evaluation with ChatGPT
│ ├── maindeepseek.py # Evaluation with DeepSeek
│ ├── reasoningandnonreasoning.py # Reasoning vs non-reasoning analysis
│ ├── semantic_fidelity.py # Semantic fidelity evaluation
│ ├── stats_robustness.py # Statistical robustness analysis
│
├── 📂 baseline/ # Baseline evaluations & metrics
│ ├── main.py
│ ├── mainaibugy.py
│ ├── mainbugsapproach.py
│ ├── maincompute_pyuguinmetrics.py
│ ├── maincoveragezero.py
│ ├── mainmaintainabilty.py
│ ├── mainpyuguin.py
│ ├── mainpyuguin_mutation.py
│ ├── mainstatandfunccov.py
│
├── 📂 scenariogenerated/ # Scenario generation pipeline
│ ├── main.py
│
├── .env # API keys (OpenAI, DeepSeek)
├── requirements.txt # Dependencies
├── README.md # Documentation
|
```

# 📌 Requirements
🖥️ System

    Python 3.9+

    Internet access for LLM API calls

Install them with:

pip install -r requirements.txt

## Running TestCopilot

To run TestCopilot on your dataset, use the following command:

```
Step#1
python scenariogenerated/main.py \
  --input dataset/HumanEval_Scenario_testcases.xlsx \
  --output outputs/scenarios

Step#2
python "LLM-Based Evaluation/mainchatgpt.py" \
  --input outputs/scenarios \
  --output outputs/evaluated_tests

```

