import pandas as pd

# === Mutation Generator ===
def generate_mutants(code: str):
    """Generate single-line mutants by replacing one operator per version."""
    mutations = {
        "==": "!=",
        "!=": "==",
        ">": "<",
        "<": ">",
        "+": "-",
        "-": "+",
        "True": "False",
        "False": "True"
    }
    mutants = []
    for key, value in mutations.items():
        if key in code:
            mutated = code.replace(key, value, 1)
            mutants.append(mutated)
    return mutants

# === Test Runner ===
def evaluate_code(code: str, test_code: str) -> int:
    """Returns 0 if test passed, 1 if test failed."""
    try:
        env = {}
        exec(code, env)
        exec(test_code, env)
        return 0  # Passed
    except Exception as e:
        return 1  # Failed

# === Main Analysis Function ===
def analyze_bug_detection_and_false_alarms(df, code_column="code", test_column="generated test cases"):
    bug_detected_count = 0
    false_alarm_count = 0
    failed_tests_on_fixed = []
    total = len(df)

    for idx, row in df.iterrows():
        original_code = str(row.get(code_column, "")).strip()
        test_code = str(row.get(test_column, "")).strip()

        if not original_code or not test_code:
            continue

        # === Bug Detection via Mutation ===
        mutants = generate_mutants(original_code)
        mutant_failed = False

        for i, mutant in enumerate(mutants):
            failed = evaluate_code(mutant, test_code)
            if failed > 0:
                mutant_failed = True
                print(f"🔴 Mutant {i} failed for sample {idx}")
                break
        if mutant_failed:
            bug_detected_count += 1

        # === False Alarm Check on Original Code ===
        failed_on_fixed = evaluate_code(original_code, test_code)
        if failed_on_fixed > 0:
            false_alarm_count += 1
            failed_tests_on_fixed.append((idx, 1))
            print(f"⚠️ Test failed on fixed code for sample {idx}")

    return {
        "Total Samples": total,
        "Bugs Detected (via mutants)": bug_detected_count,
        "False Alarms (on fixed code)": false_alarm_count,
        "Failed Tests on Fixed Code (details)": failed_tests_on_fixed
    }

# === Run the Script ===
if __name__ == "__main__":
    # Load dataset — UPDATE path if needed
    df = pd.read_excel("Zerogenerated_test_cases_HumanEval.xlsx")
    df.columns = df.columns.str.lower().str.strip()

    # Run analysis
    results = analyze_bug_detection_and_false_alarms(df, code_column="code", test_column="generated test cases")

    # Show results
    print("\n✅ Final Results:")
    print(results)

