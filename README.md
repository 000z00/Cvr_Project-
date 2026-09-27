# Multi-Class Industrial Anomaly Detection Pipeline

An end-to-end unsupervised visual defect inspection framework engineered for edge deployment and local GPU workstations (NVIDIA RTX 4050, 6GB VRAM).

This system implements a multi-scale **Reverse Distillation** architecture utilizing a frozen **WideResNet-50-2** teacher model coupled with a trainable one-class student decoder, achieving a macro mean **96.78% Image AUROC** across standard industrial manufacturing inspection benchmarks (MVTec-AD and VisA).

---

## Benchmark Performance Summary

Each category was evaluated independently to eliminate cross-domain feature interference:

| Target Category | Benchmark Dataset | Defect Type | Image AUROC | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **`metal_nut`** | MVTec-AD | Bent / Structural deformation | **99.45%** | 72.73% |
| **`screw`** | MVTec-AD | Head scratch / thread damage | **93.32%** | 24.00% |
| **`pcb1`** | VisA | Solder bridge / misaligned IC | **96.89%** | 70.73% |
| **`pcb2`** | VisA | Missing component / trace defect | **96.08%** | 68.75% |
| **`pcb3`** | VisA | Scratch / broken trace | **95.10%** | 68.35% |
| **`pcb4`** | VisA | Surface flaw / component loss | **99.86%** | 92.78% |

* **Macro Mean AUROC:** **96.78%** across all 6 evaluated industrial targets.

---

## Architectural Architecture & Methodology



Input Inspection Image: 256x256]
│
▼
┌───────────────────────────────────────────┐
│ Teacher Encoder (Frozen WideResNet-50-2)  │ ──► Multi-scale features (layer1, layer2, layer3)
└───────────────────────────────────────────┘
│
▼
┌───────────────────────────────────────────┐
│     Multi-Scale Bottleneck (OC-KD)        │
└───────────────────────────────────────────┘
│
▼
┌───────────────────────────────────────────┐
│   Student Decoder (Trained on Nominals)   │ ──► Reconstructive nominal feature synthesis
└───────────────────────────────────────────┘
│
▼
[Reconstruction Error Residual Map]
│
┌────────┴────────┐
▼                 ▼
[Pixel-level Heatmap] [PASS / FAIL Decision]


- **One-Class Knowledge Distillation:** The student decoder is trained exclusively on defect-free nominal images, learning the manifold of anomaly-free components.
- **Continuous Manifold Alignment:** Overcomes the spatial and nearest-neighbor coreset clustering degradation typical of high-density repeating circuitry on printed circuit boards.
- **Hardware-Aware Design:** Fully optimized to execute within a strict 6GB VRAM envelope without out-of-memory overhead during backpropagation.

---

## Repository Structure


├── train_all_categories_rd.py   # Full benchmark automation script across all classes
├── run_inference.py             # Interactive inference & defect overlay generator
├── setup_dataset.py             # Dataset verification & formatting utility
├── demo_results/                # Metric reports and sample defect visualizations
│   ├── benchmark_summary.json   # Machine-readable performance metrics
│   ├── heatmaps/                # Side-by-side reconstruction residual visuals
│   └── live_predictions/       # Real-time PASS/FAIL inference predictions
└── README.md                    # Project documentation


---

## Quickstart & Usage

### 1. Environment Setup
```powershell
# Create and activate Python virtual environment
python -m venv cvl_env
.\cvl_env\Scripts\activate

# Install required dependencies
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
pip install anomalib opencv-python rich

