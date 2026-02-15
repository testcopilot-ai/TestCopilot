# TESTCOPILOT  
## Scenario-Driven Test Case Generation with Autonomous Agents

![Architecture](TestCopilot.PNG)

**TESTCOPILOT** is a scenario-driven, multi-agent framework for generating **semantically accurate**, **coverage-effective**, and **maintainable** Python unit tests from natural-language functional requirements and method signatures.

Unlike one-shot requirement → test translation approaches, TESTCOPILOT:

- Synthesizes **explicit execution scenarios**
- Converts scenarios into structured `pytest` test cases
- Applies a bounded **Informant + Fixer validation loop**
- Evaluates test quality via coverage, mutation effectiveness, and maintainability metrics

This repository serves as the **official replication package** for the paper:

> **Scenario-Driven Test Case Generation with Autonomous Agents**

---

# ✨ Core Contributions

TESTCOPILOT introduces two key components:

## 🔹 Scenario-driven Contextual Generation Framework (S-CGF)

- Converts abstract requirements into structured behavioral scenarios  
- Performs contextual synthesis without external RAG corpora  
- Improves semantic reasoning and oracle fidelity  

## 🔹 Scenario-Based Test Case Generation (STCG)

- Instantiates structured scenarios into executable `pytest` test files  
- Aligns test oracles with requirement semantics  
- Improves edge-case coverage and robustness  

## 🔹 Autonomous Validation Loop

- **Informant Agent**: structural and semantic consistency validation  
- **Fixer Agent**: minimal repair of invalid or incomplete test cases  
- Bounded retries (max = 3)  
- No exposure of reference (correct) implementations  

---

# 📊 Reported Performance (From Paper)

### HumanEval
- **TCE:** 99.3%
- **Coverage:** 99.5%
- **Bugs Detected:** 179
- **False Alarms:** 0
- **Maintainability Index:** 71.30%

### MBPP
- **TCE:** 99.3%
- **Coverage:** 99.5%
- **Bugs Detected:** 157
- **False Alarms:** 0
- **Maintainability Index:** 83.31%

### LeetCode
- **TCE:** 64.8%
- **Coverage:** 92.7%
- **Bugs Detected:** 178
- **False Alarms:** 43
- **Maintainability Index:** 47.5%

> Evaluation scale: **58,912 generated test cases**

---

# 🧩 Framework Overview

### 1️⃣ Zero-Shot Initial Generation  
Generate initial candidate tests from (Requirement + Signature).

### 2️⃣ Candidate Ranking  
Select promising tests using mutation-based effectiveness and coverage signals (e.g., TCE, FAR).

### 3️⃣ Contextual Scenario Synthesis (S-CGF)  
Generate and filter analogous execution scenarios to guide reasoning.

### 4️⃣ Scenario Abstraction  
Convert requirements into structured behavioral representations:

