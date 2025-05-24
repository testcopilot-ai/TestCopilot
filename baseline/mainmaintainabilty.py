import pandas as pd
import requests
import json
import tempfile
import subprocess
import os
from time import sleep
from typing import Tuple
from radon.complexity import cc_visit
from radon.metrics import mi_visit

# === CONFIGURATION ===
DEEPSEEK_API_KEY = "Add_YOUR_API"  # 🔁 Replace this!
DEEPSEEK_API_URL = "https://model url"
MODEL = "deepseek-chat"

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {model_API_KEY}"
}

# === Dataset Path ===
dataset_path = "HumanEvalu.xlsx"
output_path = "HumanEvalu_bug_maintainability_outputs.xlsx"

# === Load and Clean Dataset ===
assert os.path.exists(dataset_path), f"❌ File not found: {dataset_path}"
df = pd.read_excel(dataset_path)
df.columns = df.columns.str.strip().str.lower()  # Normalize column names
print("✅ Loaded columns:", df.columns.tolist())

# Map safe column names
description_col = [col for col in df.columns if "functional" in col][0]
testcase_col = [col for col in df.columns if "testcase" in col][0]

# === Code cleaning: remove markdown-style wrappers ===
def clean_code(code_block: str) -> str:
    if not isinstance(code_block, str):
        return ""
    lines = code_block.strip().splitlines()
    lines = [line for line in lines if not line.strip().startswith("```")]
    return "\n".join(lines).strip()

# === DeepSeek API Wrapper ===
def call_deepseek(prompt: str, temperature: float = 0.3) -> str:
    data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a software assistant that extracts requirements and writes code."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
    }
    while True:
        response = requests.post(DEEPSEEK_API_URL, headers=HEADERS, data=json.dumps(data))
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        elif response.status_code == 429:
            print("⏳ Rate limited. Retrying in 10 seconds...")
            sleep(10)
        else:
            raise RuntimeError(f"❌ API Error {response.status_code}: {response.text}")

# === Prompt Templates ===
def requirement_extraction_prompt(description: str) -> str:
    return f"""Extract the software requirements from the following functional description.

Description:
{description}

Categorize them as follows:
Functional Requirements:
- Input/Output Conditions: ...
- Expected Behavior: ...
- Edge Cases: ...

Non-Functional Requirements:
- Time Performance: ...
- Robustness: ...
- Maintainability: ...
- Reliability: ...
"""

def code_generation_prompt(description: str, requirements: str) -> str:
    return f"""You are given a software description and its extracted requirements. Generate clean, functional Python code that satisfies all these.

Description:
{description}

Requirements:
{requirements}

Only return the Python code without explanation."""

# === Execute Testcases ===
def evaluate_code_with_test(code: str, test: str) -> Tuple[int, int]:
    detected_bug = 0
    false_alarm = 0
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "test_script.py")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(code + "\n\n" + test)
            result = subprocess.run(["python", script_path], capture_output=True, timeout=5)
            if result.returncode != 0:
                detected_bug = 1
    except Exception:
        false_alarm = 1
    return detected_bug, false_alarm

# === Analyze Maintainability & Complexity ===
def evaluate_maintainability_and_complexity(code: str) -> Tuple[float, int]:
    try:
        cleaned = clean_code(code)
        complexity = sum(block.complexity for block in cc_visit(cleaned))
        maintainability = mi_visit(cleaned, False)
        return maintainability, complexity
    except Exception:
        return 0.0, 0

# === MAIN LOOP ===
results = []

for idx, row in df.iterrows():
    description = str(row[description_col])
    test_case = str(row[testcase_col])
    print(f"🔄 Processing entry {idx + 1}/{len(df)}...")

    try:
        requirements = call_deepseek(requirement_extraction_prompt(description))
        raw_code = call_deepseek(code_generation_prompt(description, requirements), temperature=0.5)
        cleaned_code = clean_code(raw_code)
        bug, false = evaluate_code_with_test(cleaned_code, test_case)
        maintainability, complexity = evaluate_maintainability_and_complexity(cleaned_code)

        results.append({
            "Description": description,
            "Extracted Requirements": requirements,
            "Generated Code": raw_code,
            "Cleaned Code": cleaned_code,
            "Detected Bug": bug,
            "False Alarm": false,
            "Maintainability Index": maintainability,
            "Cyclomatic Complexity": complexity
        })

    except Exception as e:
        print(f"⚠️ Failed at index {idx}: {e}")
        results.append({
            "Description": description,
            "Extracted Requirements": "ERROR",
            "Generated Code": "ERROR",
            "Cleaned Code": "ERROR",
            "Detected Bug": 0,
            "False Alarm": 1,
            "Maintainability Index": 0.0,
            "Cyclomatic Complexity": 0
        })

# === SAVE RESULTS ===
pd.DataFrame(results).to_excel(output_path, index=False)
print(f"\n✅ All done. Results saved to:\n{output_path}")



