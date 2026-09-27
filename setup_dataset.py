import shutil
from pathlib import Path

BASE_DIR = Path(r"C:\Users\risha\Downloads\Documents\cvr_project")
SRC_DATASET = BASE_DIR / "Dataset"
LOCAL_WORKSPACE = BASE_DIR / "unified_dataset"

if not SRC_DATASET.exists():
    raise FileNotFoundError(f"Raw dataset folder missing at: {SRC_DATASET}")

if LOCAL_WORKSPACE.exists():
    shutil.rmtree(LOCAL_WORKSPACE)

(LOCAL_WORKSPACE / "train" / "good").mkdir(parents=True, exist_ok=True)
(LOCAL_WORKSPACE / "test" / "good").mkdir(parents=True, exist_ok=True)

# 1. MVTec categories
for cat in ["metal_nut", "screw"]:
    cat_dir = SRC_DATASET / cat
    if not cat_dir.exists():
        print(f"Skipping {cat}: not found")
        continue

    for img in (cat_dir / "train" / "good").glob("*.png"):
        shutil.copy2(img, LOCAL_WORKSPACE / "train" / "good" / f"{cat}_{img.name}")

    for img in (cat_dir / "test" / "good").glob("*.png"):
        shutil.copy2(img, LOCAL_WORKSPACE / "test" / "good" / f"{cat}_{img.name}")

    for defect_folder in (cat_dir / "test").iterdir():
        if defect_folder.is_dir() and defect_folder.name != "good":
            dest_dir = LOCAL_WORKSPACE / "test" / f"{cat}_{defect_folder.name}"
            dest_dir.mkdir(parents=True, exist_ok=True)
            for img in defect_folder.glob("*.png"):
                shutil.copy2(img, dest_dir / f"{cat}_{img.name}")

# 2. VisA categories
visa_root = SRC_DATASET / "VisA_20220922"
for pcb in ["pcb1", "pcb2", "pcb3", "pcb4"]:
    pcb_images = visa_root / pcb / "Data" / "Images"
    if not pcb_images.exists():
        print(f"Skipping {pcb}: not found")
        continue

    normal_imgs = [p for p in (pcb_images / "Normal").glob("*") if p.suffix.lower() in [".jpg", ".png"]]
    split_idx = int(0.8 * len(normal_imgs))

    for img in normal_imgs[:split_idx]:
        shutil.copy2(img, LOCAL_WORKSPACE / "train" / "good" / f"{pcb}_{img.name}")

    for img in normal_imgs[split_idx:]:
        shutil.copy2(img, LOCAL_WORKSPACE / "test" / "good" / f"{pcb}_{img.name}")

    anomaly_dir = pcb_images / "Anomaly"
    dest_dir = LOCAL_WORKSPACE / "test" / f"{pcb}_anomaly"
    dest_dir.mkdir(parents=True, exist_ok=True)
    for img in anomaly_dir.glob("*"):
        if img.suffix.lower() in [".jpg", ".png"]:
            shutil.copy2(img, dest_dir / f"{pcb}_{img.name}")

print("Dataset unification finished successfully.")
