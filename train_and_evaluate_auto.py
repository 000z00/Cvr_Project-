import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import rich
import torch

from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TQDMProgressBar,
)

rich.reconfigure(force_jupyter=False)

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import ReverseDistillation


def generate_visual_heatmaps(model, test_img_paths, output_folder, max_samples=5):
    output_folder.mkdir(parents=True, exist_ok=True)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    for img_path in test_img_paths[:max_samples]:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        orig_h, orig_w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        tensor = torch.nn.functional.interpolate(tensor, size=(256, 256), mode="bilinear", align_corners=False)
        if torch.cuda.is_available():
            tensor = tensor.cuda()

        with torch.no_grad():
            output = model(tensor)
            anomaly_map = output.anomaly_map.squeeze().cpu().numpy()
            pred_score = float(output.pred_score.cpu().item())

        norm_map = ((anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8) * 255).astype(np.uint8)
        norm_map = cv2.resize(norm_map, (orig_w, orig_h))
        heatmap = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(img_bgr, 0.65, heatmap, 0.35, 0)
        label_text = f"Score: {pred_score:.3f}"
        cv2.putText(overlay, label_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

        display_orig = cv2.resize(img_bgr, (512, 512))
        display_heat = cv2.resize(heatmap, (512, 512))
        display_over = cv2.resize(overlay, (512, 512))
        combined = np.hstack([display_orig, display_heat, display_over])

        out_path = output_folder / f"eval_vis_{img_path.name}"
        cv2.imwrite(str(out_path), combined)
        print(f"  [+] Saved evaluation overlay: {out_path.name}")


def run_pipeline(category: str, max_epochs: int = 25, patience: int = 4, batch_size: int = 8):
    torch.set_float32_matmul_precision("high")

    BASE_DIR = Path(r"C:\Users\risha\Downloads\Documents\cvr_project")
    DATASET_DIR = BASE_DIR / "unified_dataset"
    OUTPUT_DIR = BASE_DIR / "results" / f"auto_run_{category}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    test_dir = DATASET_DIR / "test"
    matches = [d for d in test_dir.iterdir() if d.is_dir() and category.lower() in d.name.lower() and d.name != "good"]
    if not matches:
        raise FileNotFoundError(f"Could not find an abnormal test folder for '{category}' in {test_dir}")
    abnormal_dir = matches[0]

    print(f"\n{'='*75}")
    print(f"   STARTING AUTOMATED PIPELINE: {category.upper()}")
    print(f"   Max Epochs: {max_epochs} | Early Stopping Patience: {patience} (on train_loss_epoch)")
    print(f"   Abnormal Source: {abnormal_dir.name}")
    print(f"{'='*75}\n")

    datamodule = Folder(
        name=f"rd_{category}",
        root=DATASET_DIR,
        normal_dir="train/good",
        normal_test_dir="test/good",
        abnormal_dir=str(abnormal_dir.relative_to(DATASET_DIR)),
        test_split_mode="from_dir",
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=2,
    )

    model = ReverseDistillation(
        backbone="wide_resnet50_2",
        layers=["layer1", "layer2", "layer3"],
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(OUTPUT_DIR / "weights"),
        filename="best_model",
        save_top_k=1,
        monitor="train_loss_epoch",
        mode="min",
        save_last=True,
    )

    early_stopping_callback = EarlyStopping(
        monitor="train_loss_epoch",
        patience=patience,
        mode="min",
        verbose=True,
        min_delta=0.001,
    )

    engine = Engine(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        default_root_dir=str(OUTPUT_DIR),
        callbacks=[
            TQDMProgressBar(refresh_rate=10),
            checkpoint_callback,
            early_stopping_callback,
        ],
    )

    print(">>> [Phase 1/3] Training with dynamic patience check on train_loss_epoch...")
    engine.fit(model=model, datamodule=datamodule)

    saved_pt_path = OUTPUT_DIR / "weights" / f"{category}_reverse_distillation.pt"
    torch.save(model.state_dict(), saved_pt_path)
    print(f"\n[+] Explicit PyTorch weights saved to: {saved_pt_path}")

    print("\n>>> [Phase 2/3] Training finished/stopped. Evaluating best model weights on test split...")
    best_ckpt = checkpoint_callback.best_model_path if checkpoint_callback.best_model_path else None
    print(f"[+] Loading best checkpoint for testing: {best_ckpt}")

    test_metrics = engine.test(model=model, datamodule=datamodule, ckpt_path=best_ckpt)
    metrics_data = test_metrics[0] if isinstance(test_metrics, list) else test_metrics

    metrics_file = OUTPUT_DIR / "evaluation_results.json"
    clean_metrics = {k: float(v) for k, v in metrics_data.items()}
    with open(metrics_file, "w") as f:
        json.dump(clean_metrics, f, indent=4)
    print(f"[+] Evaluation metrics saved to: {metrics_file}")

    print("\n>>> [Phase 3/3] Generating visual defect verification heatmaps...")
    test_images = (
        list(abnormal_dir.glob("*.jpg"))
        + list(abnormal_dir.glob("*.JPG"))
        + list(abnormal_dir.glob("*.png"))
    )
    heatmaps_dir = OUTPUT_DIR / "visual_heatmaps"
    generate_visual_heatmaps(model, test_images, heatmaps_dir, max_samples=6)

    print("\n" + "="*75)
    print(f" PIPELINE COMPLETE: {category.upper()}")
    print(f" Image AUROC : {clean_metrics.get('image_AUROC', 0.0) * 100:.2f}%")
    print(f" Image F1    : {clean_metrics.get('image_F1Score', 0.0) * 100:.2f}%")
    print(f" Artifact Weights : {saved_pt_path}")
    print(f" Verification Maps: {heatmaps_dir}")
    print("="*75 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Train & Evaluation with Early Stopping")
    parser.add_argument("--category", type=str, default="pcb1", choices=["metal_nut", "screw", "pcb1", "pcb2", "pcb3", "pcb4"])
    parser.add_argument("--max_epochs", type=int, default=25, help="Ceiling for epochs before guaranteed exit")
    parser.add_argument("--patience", type=int, default=4, help="Number of stagnant epochs before early stop")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    run_pipeline(category=args.category, max_epochs=args.max_epochs, patience=args.patience, batch_size=args.batch_size)
