import pandas as pd
import ast
import os
import coverage
import runpy
import threading

# === Paths ===
DATASET_PATH = "Arch_metrics_HumanEvalu-Agent_Tools_AI_Evaluation_Results.xlsx"
OUTPUT_PATH = "Function_Statement_Coverage_API_RunPath.xlsx"
WORK_DIR = "temp_cov_runpy"
os.makedirs(WORK_DIR, exist_ok=True)

# === Load dataset ===
df = pd.read_excel(DATASET_PATH)
df.columns = df.columns.str.strip().str.lower()

# === Results containers ===
func_cov_list = []
stmt_cov_list = []
errors = []

# === Inject test function call ===
def inject_test_call(test_code):
    try:
        tree = ast.parse(test_code)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test"):
                return f"{test_code}\n\n{node.name}()"
    except:
        pass
    return test_code

# === Timeout decorator ===
class TimeoutException(Exception):
    pass

def timeout(seconds):
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = [TimeoutException()]  # Default to TimeoutException
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    result[0] = e
            thread = threading.Thread(target=target)
            thread.start()
            thread.join(timeout=seconds)
            if thread.is_alive():
                raise TimeoutException("Timed out!")
            if isinstance(result[0], Exception):
                raise result[0]
            return result[0]
        return wrapper
    return decorator

# === Process each row ===
for idx, row in df.iterrows():
    print(f"\n🔍 Processing row {idx}...")

    code = str(row.get("code", "")).strip()
    test = str(row.get("test case", "")).strip()

    if not code or not test:
        print("❌ Skipped: missing code or test")
        func_cov_list.append(None)
        stmt_cov_list.append(None)
        errors.append("Missing code or test")
        continue

    print(f"Code:\n{code}")
    print(f"Test Case:\n{test}")

    # Inject test call if needed
    full_test = inject_test_call(test)
    merged_code = f"{code}\n\n{full_test}"
    script_path = os.path.join(WORK_DIR, f"code_{idx}.py")

    with open(script_path, "w", encoding="utf-8") as f:
        f.write(merged_code)

    try:
        @timeout(10)  # Set a timeout of 10 seconds
        def run_script():
            # Start coverage tracking
            cov = coverage.Coverage()
            cov.start()

            # Run the file as a script
            runpy.run_path(script_path, run_name="__main__")

            cov.stop()
            cov.save()
            return cov

        cov = run_script()

        analysis = cov.analysis2(script_path)
        executed_lines = set(analysis[2])
        total_lines = set(analysis[1])

        print(f"Executable lines: {analysis[1]}")
        print(f"Executed lines: {analysis[2]}")

        # Calculate statement coverage
        stmt_cov = 100 * len(executed_lines) / len(total_lines) if total_lines else None
        print(f"✅ Statement Coverage: {stmt_cov:.2f}%")

        # Analyze functions via AST
        with open(script_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())

        total_funcs = 0
        hit_funcs = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                total_funcs += 1
                lines = set(range(node.lineno, getattr(node, "end_lineno", node.lineno + 1)))
                if lines & executed_lines:
                    hit_funcs += 1
                print(f"Function: {node.name} at lines {node.lineno} to {getattr(node, 'end_lineno', node.lineno + 1)}")

        func_cov = 100 * hit_funcs / total_funcs if total_funcs > 0 else None
        print(f"✅ Function Coverage: {func_cov:.2f}%")

        func_cov_list.append(func_cov)
        stmt_cov_list.append(stmt_cov)
        errors.append(None)

    except TimeoutException as e:
        print(f"❌ Timeout error in row {idx}: {e}")
        func_cov_list.append(None)
        stmt_cov_list.append(None)
        errors.append("Timeout error")
    except Exception as e:
        print(f"❌ Error in row {idx}: {e}")
        func_cov_list.append(None)
        stmt_cov_list.append(None)
        errors.append(str(e))

# === Save results ===
df["Function Coverage (%)"] = func_cov_list
df["Statement Coverage (%)"] = stmt_cov_list
df["Coverage Error"] = errors
df.to_excel(OUTPUT_PATH, index=False)

print(f"\n✅ Final coverage results saved to:\n{OUTPUT_PATH}")
