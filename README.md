# High-Resolution Industrial Defect Detection via Multi-Scale Reverse Distillation

An industrial-grade, unsupervised visual surface inspection framework designed for edge and workstation deployment under strict 6GB VRAM hardware limits (NVIDIA GeForce RTX 4050 Laptop GPU). 

This project addresses the real-world manufacturing challenge where defect samples are rare, unpredictable, or dangerous to generate. By framing defect detection as a **One-Class Knowledge Distillation (OC-KD)** problem rather than standard feature-matching, this pipeline eliminates the memory overhead and spatial degradation associated with nearest-neighbor coreset algorithms, achieving a macro mean **96.78% Image AUROC** across 6 complex industrial inspection classes from the **MVTec-AD** and **VisA** datasets.

---

## 1. Project Motivation & Theoretical Foundation

### 1.1 The Failure Mode of Traditional Memory-Bank / PatchCore Methods
Classical unsupervised inspection pipelines (such as PatchCore) extract multi-scale patch embeddings and store them inside an external memory bank. At inference time, anomaly detection relies on $k$-Nearest Neighbors ($k$-NN) search:

$$\mathcal{A}(p) = \min_{m \in \mathcal{M}} \Vert{} \phi(p) - m \Vert{}_2$$

While effective on uniform textures, this approach exhibits severe bottlenecks on complex industrial targets:
* **High Memory Footprint:** Storing coreset patch features across multi-scale hierarchies for high-resolution images rapidly exceeds 6GB VRAM.
* **Loss of Spatial Regularization on Dense Structures:** Printed Circuit Boards (PCBs) feature high-frequency, repetitive topological traces, solder pads, and surface-mount components. Nearest-neighbor matching often matches a shifted trace to an identical trace nearby, missing subtle bridge shorts or broken pathways.

### 1.2 The Reverse Distillation Paradigm
To overcome these limitations, this pipeline utilizes **Reverse Knowledge Distillation**:

```text
                                  ┌───────────────────────────┐
                                  │   Test Image (256x256)    │
                                  └─────────────┬─────────────┘
                                                │
                                                ▼
                         ┌──────────────────────────────────────────────┐
                         │   Teacher Encoder (WideResNet-50-2 Frozen)   │
                         │   - Pretrained on ImageNet-1k                │
                         │   - Weights completely frozen during train   │
                         └───────┬──────────────┬──────────────┬────────┘
                                 │              │              │
                       Layer 1   │      Layer 2 │      Layer 3 │
                       Features  ▼     Features ▼     Features ▼
                         ┌──────────────────────────────────────────────┐
                         │   One-Class Bottleneck Embedding (OC-BE)     │
                         │   - Multi-scale feature compression          │
                         │   - Discards high-level background clutter   │
                         └──────────────────────┬───────────────────────┘
                                                │
                                                ▼
                         ┌──────────────────────────────────────────────┐
                         │     Student Decoder (Trainable Network)      │
                         │   - Symmetrical deconvolutional architecture │
                         │   - Reconstructs Layer 1, 2, 3 representations│
                         └───────┬──────────────┬──────────────┬────────┘
                                 │              │              │
                                 ▼              ▼              ▼
                         ┌──────────────────────────────────────────────┐
                         │    Cosine Distance Residual Formulation      │
                         │    S_k(h, w) = 1 - <F_T^k, F_S^k>            │
                         └──────────────────────┬───────────────────────┘
                                                │
                                                ▼
                               ┌─────────────────────────────────┐
                               │  Bilinear Upsampling & Fusion   │
                               │  Anomaly Heatmap (Continuous)   │
                               └────────────────┬────────────────┘
                                                │
                                  ┌─────────────┴─────────────┐
                                  ▼                           ▼
                        [Localized Pixel Map]       [Image PASS / FAIL Score]
```

* **The Asymmetric Information Principle:** The teacher network is a deep feature extractor that knows how to represent all visual primitives (textures, edges, defects, normal pads). The student decoder, conversely, is trained **only on normal, fault-free industrial images**.
* **Reconstruction Breakdown:** When normal components pass through, the student accurately reconstructs the teacher's feature maps, yielding near-zero cosine distance. When an anomaly (a crack, solder bridge, or missing chip) passes through, the teacher extracts rich defect descriptors, but the student lacks the parameters to reconstruct the defect manifold—yielding a high spatial residual error.

---

## 2. Quantitative Benchmark Results

Every target category was evaluated independently to evaluate category-specific decision thresholds and avoid cross-domain feature contamination:

| Target Category | Benchmark Source | Core Anomaly Characteristics | Image AUROC | Image F1-Score | Optimal Threshold |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`metal_nut`** | MVTec-AD | Structural deformations, broken threading, bent contours | **99.45%** | **72.73%** | 0.4812 |
| **`screw`** | MVTec-AD | Micro-scratches on head, damaged tip, thread flaws | **93.32%** | **24.00%** | 0.5104 |
| **`pcb1`** | VisA | Solder bridges, misaligned micro-ICs, missing SMD parts | **96.89%** | **70.73%** | 0.4635 |
| **`pcb2`** | VisA | Missing surface components, damaged copper traces | **96.08%** | **68.75%** | 0.4720 |
| **`pcb3`** | VisA | Deep surface scratches, fractured routing channels | **95.10%** | **68.35%** | 0.4590 |
| **`pcb4`** | VisA | Severe burn flaws, missing chips, pad contamination | **99.86%** | **92.78%** | 0.4910 |

* **Macro Mean Image AUROC:** **96.78%** across all evaluated industrial classes.
* **Macro Mean Inference Latency:** ~18.2 ms per $256 \times 256$ frame on NVIDIA RTX 4050.

---

## 3. Directory Layout & File Architecture

```text
cvr_project/
├── cvl_env/                              # Isolated Python 3.10 virtual environment
├── Dataset/                              # Source storage for raw uncompressed MVTec and VisA archives
├── demo_results/                         # Staged visual proof & metrics tracked for Git version control
│   ├── benchmark_summary.json            # Machine-parsable JSON report containing all test metrics
│   ├── heatmaps/                         # 3-panel comparative heatmaps separated by category
│   └── live_predictions/                # Output directory for run_inference.py predictions
├── results/                              # Runtime artifact repository (excluded from Git push)
│   ├── auto_run_<category>/              # Category run containers
│   │   ├── weights/                      # Best saved model checkpoints (.ckpt and raw .pt)
│   │   ├── visual_heatmaps/              # Dynamic evaluation residual outputs
│   │   └── evaluation_results.json       # Exact quantitative test split scores
│   └── rd_full_benchmark/                # Baseline benchmark logs and aggregated outputs
├── unified_dataset/                      # Production dataset structure consumed by Anomalib datamodules
│   ├── train/
│   │   └── good/                         # Defect-free reference imagery exclusively
│   └── test/
│       ├── good/                         # Held-out nominal test imagery
│       ├── pcb1_anomaly/                 # Categorized defect targets
│       ├── pcb2_anomaly/
│       ├── pcb3_anomaly/
│       ├── pcb4_anomaly/
│       ├── metal_nut_anomaly/
│       └── screw_anomaly/
├── .gitignore                            # Critical filter blocking model weights (>100MB) & datasets
├── cvr.ipynb                             # Interactive prototyping Jupyter Notebook
├── setup_dataset.py                      # Automated dataset standardization and folder rebuilder
├── train_pcb1_rd.py                      # Prototype reverse distillation implementation for PCB1
├── train_all_categories_rd.py            # Baseline batch training suite across all classes
├── train_and_evaluate_auto.py            # Automated training engine with EarlyStopping (patience) & eval
├── run_entire_dataset.py                 # Master orchestrator running the pipeline across all 6 classes
├── run_inference.py                      # CLI tool for single/batch image inference & heatmap overlay
└── README.md                             # Comprehensive technical documentation
```

---

## 4. In-Depth Breakdown of Every Project File

### `setup_dataset.py`
* **Role:** Dataset formatting and integrity normalization.
* **What It Does:** Raw downloads of MVTec-AD and VisA contain differing directory layouts (e.g., VisA uses flat CSV label lists while MVTec uses hierarchical folders). This script traverses the raw `Dataset/` tree, filters normal samples into `unified_dataset/train/good/`, segregates evaluation nominals into `unified_dataset/test/good/`, and groups distinct anomalies into corresponding `test/<category>_anomaly/` directories.
* **Why It Matters:** Enables the `Folder` datamodule to construct balanced evaluation splits without manual file moving.

### `train_and_evaluate_auto.py`
* **Role:** The core automated training engine.
* **What It Does:**
  1. Configures the `Folder` datamodule targeting specific defect categories with pinned GPU memory.
  2. Instantiates `ReverseDistillation` with a frozen `wide_resnet50_2` backbone tapping feature extracts at `layer1`, `layer2`, and `layer3`.
  3. Attaches an `EarlyStopping` callback monitoring `train_loss_epoch` with `patience=4` and `min_delta=0.001`. If the student decoder stops improving for 4 consecutive epochs, training terminates to prevent memorization.
  4. Attaches a `ModelCheckpoint` callback preserving only the top-performing model state (`best_model.ckpt`) and explicitly saves a raw PyTorch state dictionary (`<category>_reverse_distillation.pt`).
  5. Automatically executes `engine.test()` on the optimal checkpoint immediately after training stops—without human intervention.
  6. Dumps the evaluation metrics to `evaluation_results.json` and renders 3-panel visual error heatmaps into `visual_heatmaps/`.

### `run_entire_dataset.py`
* **Role:** Batch execution controller.
* **What It Does:** Spawns sequential subprocesses executing `train_and_evaluate_auto.py` across `metal_nut`, `screw`, `pcb1`, `pcb2`, `pcb3`, and `pcb4`.
* **Why It Matters:** Allows the entire 6-class benchmark to be trained, stopped, evaluated, and saved unattended. If any run encounters an issue, the script catches the exit code, logs it, and continues to the next category without crashing the entire queue.

### `run_inference.py`
* **Role:** Standalone production inspection utility.
* **What It Does:** Loads a trained checkpoint, accepts an arbitrary inspection image path (or a category name), scales and normalizes the tensor to $256 \times 256$, executes forward inference through the teacher-decoder pair, computes the cosine anomaly distance, applies a threshold to generate a binary PASS/FAIL decision, and exports a 3-panel visualization:
  * **Panel 1 (Raw Input):** The source component under test.
  * **Panel 2 (Jet Heatmap):** The continuous spatial cosine anomaly score map.
  * **Panel 3 (Industrial Overlay):** The raw image fused with the jet heatmap ($65\% / 35\%$ alpha blend) with anomaly score text.

### `train_all_categories_rd.py` & `train_pcb1_rd.py`
* **Role:** Foundational prototyping scripts.
* **What It Does:** The experimental scripts used during initial tuning to determine optimal learning rates, batch sizes (settling on batch size 8 for 6GB VRAM safety), and layer selection before the automated patience callbacks were developed.

### `cvr.ipynb`
* **Role:** Research notebook.
* **What It Does:** Contains exploratory code blocks evaluating feature representations, testing OpenCV normalization algorithms, and experimenting with intermediate cosine similarity operations.

---

## 5. Mathematical Mechanics of the Pipeline

### Step 1: Multi-Scale Feature Extraction
Given an input sample $I \in \mathbb{R}^{3 \times 256 \times 256}$, the frozen teacher network $T$ extracts hierarchical representations at three stages:

$$\Phi_T(I) = \{ F_T^1, F_T^2, F_T^3 \}$$

where $F_T^k \in \mathbb{R}^{C_k \times H_k \times W_k}$ represents the feature tensor from residual stage $k \in \{1, 2, 3\}$.

### Step 2: Feature Bottleneck & Student Reconstruction
The multi-scale features are compressed through a one-class bottleneck embedding module to remove noise and enforce a shared latent space. The student decoder $S$ uses symmetrical transpose convolutions to reconstruct the original multi-scale feature representations from the compressed latent code:

$$\Phi_S(I) = \{ F_S^1, F_S^2, F_S^3 \}$$

### Step 3: Cosine Distance Anomaly Map Generation
During inference, the pixel-level anomaly score map at stage $k$ is calculated by the cosine distance between the teacher's feature vector and the student's reconstructed vector at each spatial position $(h, w)$:

$$S_k(h, w) = 1 - \frac{\langle F_T^k(h, w), F_S^k(h, w) \rangle}{\Vert{} F_T^k(h, w) \Vert{}_2 \cdot \Vert{} F_S^k(h, w) \Vert{}_2}$$

### Step 4: Multi-Scale Aggregation & Upsampling
The multi-scale score maps are bilinearly interpolated back to the original image dimensions $(H, W)$ and fused:

$$\mathcal{M}_{final}(x, y) = \prod_{k=1}^3 \text{BilinearUpsample}\Big(S_k(h, w)\Big)$$

An aggregate image-level score $\mathcal{S}_{img}$ is then derived from the maximum spatial anomaly intensity:

$$\mathcal{S}_{img} = \max_{(x, y)} \mathcal{M}_{final}(x, y)$$

If $\mathcal{S}_{img} \ge \tau$, the part is marked as **FAIL (Defective)**; otherwise, it is classified as **PASS (Nominal)**.

---

## 6. Installation & Execution Guide

### 6.1 Environment Setup
Launch PowerShell in your project directory:

```powershell
# Create dedicated Python environment
python -m venv cvl_env

# Activate environment
.\cvl_env\Scripts\activate

# Install PyTorch with CUDA 12.1 acceleration
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

# Install core framework dependencies
pip install anomalib opencv-python rich
```

### 6.2 Dataset Reorganization
To structure the raw datasets into the unified layout:
```powershell
python setup_dataset.py
```

### 6.3 Train and Evaluate a Single Target
To train a single category with automated early stopping and immediate evaluation:
```powershell
python train_and_evaluate_auto.py --category pcb1 --patience 4 --max_epochs 25
```
*Options for `--category`:* `metal_nut`, `screw`, `pcb1`, `pcb2`, `pcb3`, `pcb4`.

### 6.4 Run the Full Benchmark Suite (Unattended Batch)
To run all 6 categories consecutively:
```powershell
python run_entire_dataset.py
```
Each category will automatically train until the loss plateaus, save its weights, execute testing, generate visual overlays, and proceed to the next class.

### 6.5 Run Interactive Single-Image Inference
To run a sample check on an arbitrary image:
```powershell
# Run using the built-in test set for a category
python run_inference.py --category pcb4

# Run against a specific external test image
python run_inference.py --category metal_nut --image_path "C:\path\to\damaged_nut.png"
```
The resulting heatmaps and classification decisions are written directly to `demo_results/live_predictions/`.
