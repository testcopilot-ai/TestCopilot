#scenario generation and Scenario generated test cases 
import openai
import pandas as pd
import json
import ast
from sentence_transformers import SentenceTransformer, util

# === API Key ===
openai.api_key = "your-openai-api-key"

# === Load your Excel File ===
input_file = "functional_signatures.xlsx"
df = pd.read_excel(input_file)
df.columns = df.columns.str.strip().str.lower()

# === SRAG Knowledge Base ===
past_requirements = [
    "Check if a number is prime",
    "Determine whether two strings are anagrams",
    "Return true if the list is sorted",
    "Check if the input string reads the same backward",
    "Determine whether a number is even or odd",
    "Find maximum element in a list",
    "Verify if parentheses are balanced",
    "Compute factorial of a number"
]

# === Load SentenceTransformer ===
model = SentenceTransformer('all-MiniLM-L6-v2')

# === Output Lists ===
scenario_records = []
testcase_records = []

# === Loop Through Each Functional Requirement ===
for idx, row in df.iterrows():
    func_req = str(row['functional_requirement']).strip()
    signature = str(row['signature']).strip()
    print(f"\n🔍 Processing: {func_req}")

    # --- SRAG Retrieval ---
    embeddings = model.encode([func_req] + past_requirements, convert_to_tensor=True)
    query_embedding = embeddings[0]
    similarities = util.pytorch_cos_sim(query_embedding, embeddings[1:])[0]
    top_indices = similarities.topk(3).indices
    examples = [past_requirements[i] for i in top_indices]

    # --- Generate Scenarios ---
    prompt_scenarios = f"""You are given a function signature and its functional requirement.

Function Signature:
{signature}

Functional Requirement:
{func_req}

Here are similar examples:
- {examples[0]}
- {examples[1]}
- {examples[2]}

Based on this, generate 3 structured scenarios. Each scenario must include:
- input
- expected output
- postcondition

Return the result as a JSON list with keys: 'input', 'output', 'postcondition'."""
    
    response_scenarios = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt_scenarios}],
        temperature=0.4
    )
    
    try:
        scenarios = json.loads(response_scenarios['choices'][0]['message']['content'])
    except Exception:
        scenarios = ast.literal_eval(response_scenarios['choices'][0]['message']['content'])

    # --- Generate Test Cases ---
    prompt_tests = f"""You are given the following function signature:

{signature}

And the following test scenarios in JSON:

{json.dumps(scenarios, indent=2)}

Write one unittest test method in Python per scenario. Return only the test methods without imports or boilerplate."""
    
    response_tests = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt_tests}],
        temperature=0.3
    )

    test_methods = response_tests['choices'][0]['message']['content'].strip().split('\n\n')

    # --- Store Records ---
    for i, scenario in enumerate(scenarios):
        scenario_records.append({
            "row_id": idx + 1,
            "functional_requirement": func_req,
            "signature": signature,
            "scenario_id": f"{idx + 1}-{i + 1}",
            "input": scenario['input'],
            "output": scenario['output'],
            "postcondition": scenario['postcondition']
        })
        testcase_records.append({
            "scenario_id": f"{idx + 1}-{i + 1}",
            "test_case": test_methods[i] if i < len(test_methods) else "N/A"
        })

# === Save to Excel ===
df_scenarios = pd.DataFrame(scenario_records)
df_tests = pd.DataFrame(testcase_records)

output_file = "TestCopilot_Generated_Scenarios_TestCases.xlsx"
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    df_scenarios.to_excel(writer, sheet_name="Scenarios", index=False)
    df_tests.to_excel(writer, sheet_name="TestCases", index=False)

print(f"\n✅ All scenarios and test cases saved to '{output_file}'")

