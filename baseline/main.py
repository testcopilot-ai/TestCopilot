import pandas as pd
import requests
import json

def read_excel_data(file_path):
    """Read function details from an Excel file."""
    data = pd.read_excel(file_path)
    data.columns = data.columns.str.strip()  # Remove extra spaces
    return data

def generate_test_cases_with_deepseek(api_key, function_signature, functional_requirement):
    """Generate test cases using the API."""
    api_url = "https://apiurl"DeepSeek, chatgpt""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    prompt = f"""
    Generate multiple test cases (at least 5) for the following function, covering normal and edge cases.

    **Function Signature:**
    {function_signature}

    **Functional Requirement:**
    {functional_requirement}

    Return the test cases in Python `assert` format.
    """

    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 1000
    }

    response = requests.post(api_url, headers=headers, data=json.dumps(payload))
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"API Error: {response.status_code} - {response.text}")

def main():
    # Replace with your actual  API key
    api_key = "Add your api key"
    input_excel = "HumanEvaluZero.xlsx"  # Your input file
    output_excel = "Zerogenerated_test_cases_HumanEval.xlsx"  # Output file

    try:
        data = read_excel_data(input_excel)
    except Exception as e:
        print(f"❌ Error reading Excel file: {e}")
        return

    # Use YOUR column names (case-sensitive)
    required_columns = ["functional_requirement", "Function signature"]
    available_columns = list(data.columns)

    missing_columns = [col for col in required_columns if col not in available_columns]
    if missing_columns:
        print(f"❌ Missing columns: {missing_columns}")
        print(f"Available columns: {available_columns}")
        return

    test_cases = []
    for _, row in data.iterrows():
        signature = row["Function signature"]  # Match exact column name
        requirement = row["functional_requirement"]  # Match exact column name

        try:
            generated_tests = generate_test_cases_with_deepseek(api_key, signature, requirement)
            test_cases.append(generated_tests)
        except Exception as e:
            print(f"❌ Error generating test cases for: {signature} - {e}")
            test_cases.append("Failed to generate test cases")

    data["Generated Test Cases"] = test_cases

    try:
        data.to_excel(output_excel, index=False)
        print(f"✅ Success! Test cases saved to: {output_excel}")
    except Exception as e:
        print(f"❌ Error saving results: {e}")

if __name__ == "__main__":
    main()
