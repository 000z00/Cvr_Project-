import rich
from pathlib import Path
import torch

rich.reconfigure(force_jupyter=False)

from anomalib.data import Folder
from anomalib.models import ReverseDistillation
from anomalib.engine import Engine
from lightning.pytorch.callbacks import TQDMProgressBar

def main():
    torch.set_float32_matmul_precision("high")
    
    BASE_DIR = Path(r"C:\Users\risha\Downloads\Documents\cvr_project\unified_dataset")
    OUTPUT_DIR = Path(r"C:\Users\risha\Downloads\Documents\cvr_project\results\rd_pcb1")

    # Ingest PCB1 from unified_dataset
    datamodule = Folder(
        name="visa_pcb1",
        root=BASE_DIR,
        normal_dir="train/good",
        normal_test_dir="test/good",
        abnormal_dir="test/pcb1_anomaly",
        test_split_mode="from_dir",
        train_batch_size=8,
        eval_batch_size=8,
        num_workers=2,
    )

    # Reverse Distillation: WideResNet-50-2 Teacher with Multi-Scale Decoder
    model = ReverseDistillation(
        backbone="wide_resnet50_2",
        layers=["layer1", "layer2", "layer3"],
    )

    # 10 epochs of teacher-decoder feature alignment
    engine = Engine(
        max_epochs=10,
        accelerator="auto",
        devices=1,
        default_root_dir=str(OUTPUT_DIR),
        callbacks=[TQDMProgressBar(refresh_rate=10)]
    )

    print("--- Training Reverse Distillation (WideResNet-50) on PCB1 ---")
    engine.fit(model=model, datamodule=datamodule)
    
    print("\n--- Evaluating Reverse Distillation on PCB1 ---")
    metrics = engine.test(model=model, datamodule=datamodule)
    print("\n================ Reverse Distillation PCB1 Metrics ================")
    print(metrics)
    print("===================================================================")

if __name__ == "__main__":
    main()
