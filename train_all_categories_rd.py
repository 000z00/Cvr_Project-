import cv2
import json
import torch
import numpy as np
from pathlib import Path
import rich
from lightning.pytorch.callbacks import TQDMProgressBar

rich.reconfigure(force_jupyter=False)

from anomalib.data import Folder
from anomalib.models import ReverseDistillation
from anomalib.engine import Engine

CATEGORIES = [
    {"name": "metal_nut", "dataset": "mvtec", "abnormal_dir": "test/metal_nut_bent"},
    {"name": "screw",     "dataset": "mvtec", "abnormal_dir": "test/screw_scratch_head"},
    {"name": "pcb1",      "dataset": "visa",  "abnormal_dir": "test/pcb1_anomaly"},
    {"name": "pcb2",      "dataset": "visa",  "abnormal_dir": "test/pcb2_anomaly"},
    {"name": "pcb3",      "dataset": "visa",  "abnormal_dir": "test/pcb3_anomaly"},
    {"name": "pcb4",      "dataset": "visa",  "abnormal_dir": "test/pcb4_anomaly"},
]

def generate_visual_heatmaps(model, test_img_paths, output_folder):
    output_folder.mkdir(parents=True, exist_ok=True)
    model.eval().cuda()
    
    for img_path in test_img_paths[:4]:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        tensor = torch.nn.functional.interpolate(tensor, size=(256, 256), mode="bilinear", align_corners=False).cuda()
        
        with torch.no_grad():
            output = model(tensor)
            anomaly_map = output.anomaly_map.squeeze().cpu().numpy()
            
        norm_map = ((anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8) * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
        resized_orig = cv2.resize(img_bgr, (256, 256))
        overlay = cv2.addWeighted(resized_orig, 0.6, heatmap, 0.4, 0)
        combined = np.hstack([resized_orig, heatmap, overlay])
        
        cv2.imwrite(str(output_folder / f"vis_{img_path.name}"), combined)

def main():
    torch.set_float32_matmul_precision("high")
    
    WORKSPACE = Path(r"C:\Users\risha\Downloads\Documents\cvr_project\unified_dataset")
    BASE_OUTPUT = Path(r"C:\Users\risha\Downloads\Documents\cvr_project\results\rd_full_benchmark")
    
    summary_results = {}
    
    for cat in CATEGORIES:
        name = cat["name"]
        print(f"\n{'='*70}")
        print(f"   STARTING REVERSE DISTILLATION PIPELINE: {name.upper()}")
        print(f"{'='*70}\n")
        
        cat_output_dir = BASE_OUTPUT / name
        heatmaps_dir = BASE_OUTPUT / "heatmaps" / name
        
        # Verify abnormal path exists, fallback to generic pattern if named differently
        abnormal_path = WORKSPACE / cat["abnormal_dir"]
        if not abnormal_path.exists():
            matches = list((WORKSPACE / "test").glob(f"*{name}*"))
            valid_dirs = [d for d in matches if d.is_dir() and d.name != "good"]
            abnormal_path = valid_dirs[0] if valid_dirs else (WORKSPACE / "test")

        datamodule = Folder(
            name=f"rd_{name}",
            root=WORKSPACE,
            normal_dir="train/good",
            normal_test_dir="test/good",
            abnormal_dir=str(abnormal_path.relative_to(WORKSPACE)),
            test_split_mode="from_dir",
            train_batch_size=8,
            eval_batch_size=8,
            num_workers=2,
        )

        model = ReverseDistillation(
            backbone="wide_resnet50_2",
            layers=["layer1", "layer2", "layer3"],
        )

        engine = Engine(
            max_epochs=10,
            accelerator="auto",
            devices=1,
            default_root_dir=str(cat_output_dir),
            callbacks=[TQDMProgressBar(refresh_rate=10)]
        )

        # Train model
        torch.cuda.empty_cache()
        engine.fit(model=model, datamodule=datamodule)
        
        # Evaluate model
        metrics = engine.test(model=model, datamodule=datamodule)
        metric_res = metrics[0] if isinstance(metrics, list) else metrics
        
        summary_results[name] = {
            "image_AUROC": float(metric_res.get("image_AUROC", 0.0)),
            "image_F1Score": float(metric_res.get("image_F1Score", 0.0)),
            "dataset": cat["dataset"]
        }

        # Generate sample heatmaps
        test_images = list(abnormal_path.glob("*.jpg")) + list(abnormal_path.glob("*.JPG")) + list(abnormal_path.glob("*.png"))
        if test_images:
            generate_visual_heatmaps(model, test_images, heatmaps_dir)
            print(f"Heatmaps exported to: {heatmaps_dir}")

    # Print and save summary
    print("\n\n" + "="*70)
    print("        REVERSE DISTILLATION (WIDE-RESNET-50) BENCHMARK REPORT")
    print("="*70)
    print(f"{'Category':<15} | {'Dataset':<10} | {'Image AUROC':<15} | {'F1-Score':<10}")
    print("-"*70)
    for name, res in summary_results.items():
        print(f"{name:<15} | {res['dataset']:<10} | {res['image_AUROC']*100:>10.2f}%    | {res['image_F1Score']*100:>8.2f}%")
    print("="*70)

    with open(BASE_OUTPUT / "benchmark_summary.json", "w") as f:
        json.dump(summary_results, f, indent=4)
    print(f"\nComplete benchmark metrics saved to: {BASE_OUTPUT / 'benchmark_summary.json'}")

if __name__ == "__main__":
    main()
