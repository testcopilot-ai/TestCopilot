import openai
import pandas as pd

# === CONFIG ===
openai.api_key = "your-openai-api-key"  # Replace with your OpenAI key
input_file = "functional_signatures.xlsx"  # Your input Excel file
output_file = "generated_test_cases.xlsx"  # Output Excel file

# === Read Excel File ===
df = pd.read_excel(input_file)
df.columns = df.columns.str.strip().str.lower()

# === Output Storage ===
results = []

# === Loop Through Each Row ===
for idx, row in df.iterrows():
    func_req = str(row['functional_requirement']).strip()
    signature = str(row['signature']).strip()

    print(f"\n🔍 Generating test cases for: {signature} - {func_req}")

    # === Prompt to LLM ===
    prompt = f"""
You are given the following Python function signature and its functional requirement.

Function Signature:
{signature}

Functional Requirement:
{func_req}

Please generate Python unit test cases using unittest format. 
Only return the test method code (without imports or main block).
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        test_code = response['choices'][0]['message']['content'].strip()
    except Exception as e:
        test_code = f"Error: {str(e)}"

    # === Save result ===
    results.append({
        "functional_requirement": func_req,
        "signature": signature,
        "test_case": test_code
    })

# === Save to Excel ===
df_out = pd.DataFrame(results)
df_out.to_excel(output_file, index=False)

print(f"\n✅ All test cases saved to: {output_file}")

