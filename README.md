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
DDP (HumanEval)	 99.7%
Function Coverage	89.5%
Bugs Detected (HumanEval)	179
False Alarms	0
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
📦 TestCopilot/
│
├── 📂 dataset/                  # HumanEval / MBPP test scenario datasets
│   ├── HumanEval_Scenario_testcases.xlsx
│   ├── MBPP_Scenario_testcases.xlsx
│
├── 📂 LLM-Based Evaluation/                   # Multi-agent modules
│   ├── mainchatgpt.py            # Test evaluation and test fixer agent
│   ├── maindeepseek.py 
│
├── 📂 Baseline/              # Evaluation and metric calculation
│   ├── coverage_metrics.py
│   ├── ddp_tce_metrics.py
│   ├── maintainability.py
│
├── 📂 ScenarioGeneration/              # Scenario generation
│   ├── main.py
|
├── .env                        # API Keys (OpenAI, DeepSeek)
├── requirements.txt           # Required packages
├── README.md                  # Documentation file
```

# 📌 Requirements
🖥️ System

    Python 3.9+

    Internet access for LLM API calls

Install them with:

pip install -r requirements.txt

## Running TestCopilot

To run TestCopilot on your dataset, use the following command:

```bash
python scripts/run_testcopilot.py --input_dir path/to/your/input --output_dir path/to/save/results
```

