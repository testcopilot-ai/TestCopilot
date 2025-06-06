import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from openai import OpenAI

# Set API Key for openai
openai_API_KEY = "Add_your_key_here"
Openai_BASE_URL = "url"
client = OpenAI(api_key=add_keyname, base_url=openai_BASE_URL)

# Informant Agent: Verifies the completeness of inputs
def informant_agent(functional_requirement, scenario, testcase, code):
    if not all([functional_requirement, scenario, testcase, code]):
        return "Error: Missing required inputs.", False
    prompt = f"""
    Validate the following inputs:
    Functional Requirement: {functional_requirement}
    Scenario: {scenario}
    Test Case: {testcase}
    Code: {code}

    Check if all inputs are well-defined and consistent.
    """
    response = chat_with_deepseek(prompt)
    return response, "valid" in response.lower()

# Fixer Agent: Evaluates and fixes test cases
def fixer_agent(functional_requirement, scenario, testcase, code):
    evaluation_prompt = f"""
    Functional Requirement: {functional_requirement}
    Scenario: {scenario}
    Test Case: {testcase}
    Code: {code}

    Evaluate the following:
    1. Read the functional requirement, Scanerio generated by self-RAG, further look at the test cases and code.
    2. Evaluate testcases based on the functional requirement and self-RAG. Understand user intention and evaluate test cases based on the given code.
    3. Does the test case adequately cover the functional requirement?
    4. Specify what was the reason behind if the test case passes or fails.
    5. If it fails, rewrite the correct code of the failing test case based on functional requirement and and ensure
    the logic is such that it would pass the corresponding testcase provided
    4. Evaluate code coverage (branch coverage, line coverage, functional coverage, Test Case Effectiveness, Defect Detection Percentage (DDP) 
    Function Coverage, Statement Coverage, Path Coverage, Shallow Coverage, Deep Coverage, Integration Coverage).
    5. Evaluate test case effectiveness and give the total percentage of evaluation matrics above.
    6.If starting code is provided look at the issues mentioned and attempt to fix them
    7. Ensure that the signature of the function is the same as the one provided by the user
    """
    evaluation = chat_with_deepseek(evaluation_prompt)

    if "fail" in evaluation.lower():
        fix_prompt = f"""
        The following test case and code have issues. Identify the problems and provide corrected versions:

        Test Case:
        {testcase}

        Code:
        {code}

        Provide fixed test cases and code.
        """
        fixed_response = chat_with_deepseek(fix_prompt)
        return evaluation, fixed_response

    return evaluation, None

# Helper function for interacting with DeepSeek
def chat_with_deepseek(user_input, model="deepseek-chat"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_input},
            ],
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

# Compute metrics
def compute_metrics(y_true, y_pred):
    """
    Computes evaluation metrics: precision, recall, F1 score, and accuracy.
    """
    precision = precision_score(y_true, y_pred, average='binary', zero_division=1)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=1)
    accuracy = accuracy_score(y_true, y_pred)
    return precision, recall, f1, accuracy

def main():
    # Load dataset
    file_path = "mbpp_Scanerio_testcases.xlsx"
    df = pd.read_excel(file_path, sheet_name="Лист1")

    # Strip spaces from column names
    df.columns = df.columns.str.strip()

    results = []
    y_true = []
    y_pred = []

    max_retries = 3
    for index, row in df.iterrows():
        functional_requirement = row.get("functional requirement")
        scenario = row.get("Scenarios")
        testcase = row.get("Testcases")
        code = row.get("Code")

        # Informant Agent Validation
        informant_response, is_valid = informant_agent(functional_requirement, scenario, testcase, code)
        if not is_valid:
            print(f"Row {index}: {informant_response}")
            results.append({
                "Functional Requirement": functional_requirement,
                "Scenario": scenario,
                "Test Case": testcase,
                "Evaluation": informant_response,
                "Fixed Test Case": "N/A",
                "Fixed Code": "N/A",
            })
            y_pred.append(0)
            y_true.append(0)
            continue

        # Fixer Agent Evaluation
        retries = 0
        while retries < max_retries:
            evaluation, fixed_response = fixer_agent(functional_requirement, scenario, testcase, code)
            print(f"Evaluation for row {index}: {evaluation}")

            if "pass" in evaluation.lower():
                y_pred.append(1)
                y_true.append(1)
                results.append({
                    "Functional Requirement": functional_requirement,
                    "Scenario": scenario,
                    "Test Case": testcase,
                    "Evaluation": evaluation,
                    "Fixed Test Case": "N/A",
                    "Fixed Code": "N/A",
                })
                break
            else:
                y_pred.append(0)
                y_true.append(0)
                retries += 1

                results.append({
                    "Functional Requirement": functional_requirement,
                    "Scenario": scenario,
                    "Test Case": testcase,
                    "Evaluation": evaluation,
                    "Fixed Test Case": fixed_response,
                    "Fixed Code": "N/A" if not fixed_response else fixed_response,
                })

                if retries == max_retries:
                    print(f"Max retries reached for row {index}")

    # Compute metrics
    precision, recall, f1, accuracy = compute_metrics(y_true, y_pred)

    # Save results to Excel
    results_df = pd.DataFrame(results)
    output_file_path = "mbpp-Agent_Tools_AI_Evaluation_Results.xlsx"
    try:
        results_df.to_excel(output_file_path, index=False)
        print(f"Results saved to {output_file_path}")
    except Exception as e:
        print(f"Error saving results: {e}")

    # Print metrics
    print(f"Metrics: Precision={precision}, Recall={recall}, F1 Score={f1}, Accuracy={accuracy}")

if __name__ == "__main__":
    main()
