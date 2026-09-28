import subprocess
import sys
import time

CATEGORIES = ["metal_nut", "screw", "pcb1", "pcb2", "pcb3", "pcb4"]
PYTHON_EXE = sys.executable

print("=" * 80)
print("  LAUNCHING FULL DATASET BENCHMARK SUITE (6 CATEGORIES)")
print("  Architecture: Reverse Distillation (WideResNet-50-2)")
print("  Patience: 4 epochs | Max Epochs: 25")
print("=" * 80)

overall_start = time.time()
summary_status = {}

for idx, category in enumerate(CATEGORIES, 1):
    print(f"\n[{idx}/{len(CATEGORIES)}] >>> STARTING RUN FOR: {category.upper()}")
    cat_start = time.time()
    
    cmd = [
        PYTHON_EXE,
        "train_and_evaluate_auto.py",
        "--category", category,
        "--patience", "4",
        "--max_epochs", "25",
        "--batch_size", "8"
    ]
    
    result = subprocess.run(cmd)
    elapsed = (time.time() - cat_start) / 60
    
    if result.returncode == 0:
        summary_status[category] = f"SUCCESS ({elapsed:.1f} min)"
        print(f"[+] Completed {category} successfully in {elapsed:.1f} minutes.")
    else:
        summary_status[category] = f"FAILED (exit code {result.returncode})"
        print(f"[-] Category {category} encountered an error.")

total_elapsed = (time.time() - overall_start) / 60
print("\n" + "=" * 80)
print(f"  BENCHMARK SUITE COMPLETE (Total Time: {total_elapsed:.1f} min)")
print("=" * 80)
for cat, status in summary_status.items():
    print(f"  - {cat:<12}: {status}")
print("=" * 80 + "\n")
