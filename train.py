import argparse
import os
from ultralytics import YOLO
import torch

def parse_args():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser = argparse.ArgumentParser(description="YOLOv8 Training with Auto-Resume and Anti-Overfitting for Pallet & Box Detection")
    
    # Dataset & Weights
    default_data_path = os.path.join(current_dir, "data.yaml")
    parser.add_argument("--data", type=str, default=default_data_path, help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolov8m.pt", help="Base model weights")
    
    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=100, help="Total training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--patience", type=int, default=25, help="Early stopping patience")
    
    # Paths & Checkpointing
    default_project_path = os.path.join(current_dir, "runs")
    parser.add_argument("--project", type=str, default=default_project_path, help="Directory to save runs")
    parser.add_argument("--name", type=str, default="pallet_detection", help="Run name")
    parser.add_argument(
        "--save_period", 
        type=int, 
        default=1, 
        help="Save checkpoint every X epochs"
    )
    
    # Export
    parser.add_argument("--export_onnx", action="store_true", help="Export best model to ONNX after training")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Device set to: {device}")
    
    # Target checkpoint path for the current run name
    last_checkpoint_path = os.path.join(args.project, args.name, "weights", "last.pt")
    
    # AUTO-RESUME LOGIC
    if os.path.exists(last_checkpoint_path):
        print(f"\n[INFO] Found interrupted run! Auto-resuming from: {last_checkpoint_path}")
        # Load the partially trained model
        model = YOLO(last_checkpoint_path)
        # Resume exact optimizer state, learning-rate schedule, and epoch counter
        model.train(resume=True)
    else:
        print(f"\n[INFO] Starting heavily augmented training with base weights: {args.model}")
        print(f"Dataset configuration: {args.data}")
        print(f"Checkpoints will be saved every {args.save_period} epoch(s) in: {os.path.join(args.project, args.name, 'weights')}")
        
        model = YOLO(args.model)
        model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            patience=args.patience,
            project=args.project,
            name=args.name,
            device=device,
            save=True,                   
            save_period=args.save_period, 
            plots=True,
            verbose=True,
            
            # REGULARIZATION
            weight_decay=0.001,  # Stronger penalty on large weights to prevent memorization
            dropout=0.2,         # Drop 20% of neurons randomly
            
            # DATA AUGMENTATION
            hsv_h=0.02,          # Hue variation
            hsv_s=0.8,           # Saturation variation for handling warehouse lighting shifts
            hsv_v=0.6,           # Value (brightness) variation to simulate dark shadows and glare
            degrees=15.0,        # Random rotation up to 15 degrees to handle skewed pallets
            translate=0.2,       # Random translation (panning)
            scale=0.6,           # Image scale variation
            perspective=0.0001,  # Random perspective skewing to match varied floor angles
            mosaic=1.0,          # Ensure mosaic augmentation is always on
            mixup=0.1            # Mix images together 10% of the time to generalize backgrounds
        )

    # Validate best model
    print("\nStarting validation phase on best checkpoint...")
    metrics = model.val()
    print("\n--- Validation Metrics ---")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50:    {metrics.box.map50:.4f}")

    # Export to ONNX
    if args.export_onnx:
        print("\nExporting model to ONNX...")
        model.export(format="onnx", simplify=True)
        print("ONNX export complete.")

if __name__ == "__main__":
    main()