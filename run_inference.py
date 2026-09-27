import argparse
from pathlib import Path
import cv2
import numpy as np
import torch

from anomalib.models import ReverseDistillation


def load_model_from_category(category_name: str, base_results_dir: Path):
  category_dir = base_results_dir / category_name
  ckpt_files = list(category_dir.rglob("*.ckpt"))

  if not ckpt_files:
    raise FileNotFoundError(
        f"No .ckpt checkpoint found inside {category_dir}. Ensure training"
        " finished."
    )

  best_ckpt = str(ckpt_files[0])
  print(f"[+] Loading trained weights from: {best_ckpt}")

  model = ReverseDistillation.load_from_checkpoint(best_ckpt)
  model.eval()
  if torch.cuda.is_available():
    model.cuda()
  return model


def predict_single_image(
    model, image_path: Path, output_dir: Path, threshold: float = 0.5
):
  img_bgr = cv2.imread(str(image_path))
  if img_bgr is None:
    raise ValueError(f"Could not read image from {image_path}")

  orig_h, orig_w = img_bgr.shape[:2]
  img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

  # Preprocess
  tensor = (
      torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
  )
  tensor = torch.nn.functional.interpolate(
      tensor, size=(256, 256), mode="bilinear", align_corners=False
  )

  if torch.cuda.is_available():
    tensor = tensor.cuda()

  # Inference
  with torch.no_grad():
    output = model(tensor)
    anomaly_map = output.anomaly_map.squeeze().cpu().numpy()
    pred_score = float(output.pred_score.cpu().item())

  # Normalize anomaly map to 0-255
  norm_map = (
      (anomaly_map - anomaly_map.min())
      / (anomaly_map.max() - anomaly_map.min() + 1e-8)
      * 255
  ).astype(np.uint8)
  norm_map = cv2.resize(norm_map, (orig_w, orig_h))
  heatmap = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)

  # Blend overlay
  overlay = cv2.addWeighted(img_bgr, 0.65, heatmap, 0.35, 0)

  # Determine decision
  is_anomalous = pred_score >= threshold
  status_text = (
      f"FAIL (Defect: {pred_score:.3f})"
      if is_anomalous
      else f"PASS (Normal: {pred_score:.3f})"
  )
  color = (0, 0, 255) if is_anomalous else (0, 255, 0)

  cv2.putText(
      overlay,
      status_text,
      (20, 40),
      cv2.FONT_HERSHEY_SIMPLEX,
      1.0,
      color,
      2,
      cv2.LINE_AA,
  )

  # Export side-by-side visualization
  display_img = cv2.resize(img_bgr, (512, 512))
  display_heat = cv2.resize(heatmap, (512, 512))
  display_over = cv2.resize(overlay, (512, 512))
  combined = np.hstack([display_img, display_heat, display_over])

  output_dir.mkdir(parents=True, exist_ok=True)
  out_file = output_dir / f"pred_{image_path.stem}.png"
  cv2.imwrite(str(out_file), combined)

  print(f"[Result] Image: {image_path.name}")
  print(f"         Prediction: {status_text}")
  print(f"         Saved visualization to: {out_file}\n")


def main():
  parser = argparse.ArgumentParser(
      description="Run Reverse Distillation Inference"
  )
  parser.add_argument(
      "--category",
      type=str,
      default="pcb4",
      choices=["metal_nut", "screw", "pcb1", "pcb2", "pcb3", "pcb4"],
      help="Object category",
  )
  parser.add_argument(
      "--image_path",
      type=str,
      default=None,
      help="Specific test image path. If not provided, tests first abnormal sample.",
  )
  args = parser.parse_args()

  BASE_DIR = Path(r"C:\Users\risha\Downloads\Documents\cvr_project")
  RESULTS_DIR = BASE_DIR / "results" / "rd_full_benchmark"
  OUTPUT_DIR = BASE_DIR / "results" / "live_predictions" / args.category

  # Load model
  model = load_model_from_category(args.category, RESULTS_DIR)

  # Pick test image
  if args.image_path:
    test_img = Path(args.image_path)
  else:
    # Auto-find a sample from unified_dataset test folders
    test_folder = BASE_DIR / "unified_dataset" / "test"
    matches = list(test_folder.glob(f"*{args.category}*/*.[jJ][pP][gG]")) + list(
        test_folder.glob(f"*{args.category}*/*.png")
    )
    if not matches:
      raise FileNotFoundError(
          f"No test images found matching category {args.category}."
      )
    test_img = matches[0]

  predict_single_image(model, test_img, OUTPUT_DIR)


if __name__ == "__main__":
  main()