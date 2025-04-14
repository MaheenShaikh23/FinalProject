import subprocess

def run_script(script_path):
    print(f"Running {script_path}...")
    result = subprocess.run(["python", script_path], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)

if __name__ == "__main__":
    scripts = [
        #"dataset.py",
        "utils/data_preprocessing.py",
        "models/train_rf.py",
        "models/train_xgb.py",
        "models/stacking.py"
    ]

    for script in scripts:
        run_script(script)

    print("All steps completed successfully.")
