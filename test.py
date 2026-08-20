import argparse
import json
import os
from pathlib import Path
import pandas as pd
import torch
from ultralytics import YOLO


def parse_args():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description="YOLOv8 Test Evaluation and Output Saving"
    )

    # Model and Data paths
    default_weights = os.path.join(
        current_dir, "runs", "pallet_run_1", "weights", "best.pt"
    )
    default_data = os.path.join(current_dir, "data.yaml")
    default_output = os.path.join(current_dir, "runs", "test_evaluation")

    parser.add_argument(
        "--weights",
        type=str,
        default=default_weights,
        help="Path to trained best.pt weights",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=default_data,
        help="Path to data.yaml dataset config",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=default_output,
        help="Directory to store predictions and metric files",
    )

    # Hyperparameters
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Inference image resolution"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for prediction visualization",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="NMS IoU threshold for prediction visualization",
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Save predicted label bounding boxes as .txt files",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    device = "0" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.weights):
        raise FileNotFoundError(
            f"Trained model weights not found at: {args.weights}"
        )

    print("=" * 70)
    print(f"Loading Model Checkpoint: {args.weights}")
    print(f"Dataset Config:          {args.data}")
    print(f"Output Destination:      {args.output_dir}")
    print(f"Compute Device:          {device}")
    print("=" * 70)

    model = YOLO(args.weights)

    # 1. EVALUATION ON TEST SPLIT (Metric Distributions & Scores)
    print("\n[Step 1/2] Computing Quantitative Evaluation Metrics on Test Split...")
    val_results = model.val(
        data=args.data,
        split="test",  # Evaluates on the held-out test split
        imgsz=args.imgsz,
        device=device,
        plots=True,
        project=args.output_dir,
        name="test_metrics",
        exist_ok=True,
    )

    # Extract overall metrics
    overall_p = float(val_results.box.mp)
    overall_r = float(val_results.box.mr)
    overall_map50 = float(val_results.box.map50)
    overall_map50_95 = float(val_results.box.map)
    overall_f1 = (
        (2 * overall_p * overall_r) / (overall_p + overall_r + 1e-9)
        if (overall_p + overall_r) > 0
        else 0.0
    )

    # Extract per-class breakdown
    class_names = list(val_results.names.values())
    per_class_summary = []

    for idx, name in enumerate(class_names):
        p_cls = float(val_results.box.p[idx])
        r_cls = float(val_results.box.r[idx])
        map50_cls = float(val_results.box.ap50[idx])
        map50_95_cls = float(val_results.box.ap[idx])
        f1_cls = (
            (2 * p_cls * r_cls) / (p_cls + r_cls + 1e-9)
            if (p_cls + r_cls) > 0
            else 0.0
        )

        per_class_summary.append(
            {
                "Class": name,
                "Precision": round(p_cls, 4),
                "Recall": round(r_cls, 4),
                "F1-Score": round(f1_cls, 4),
                "mAP@50": round(map50_cls, 4),
                "mAP@50-95": round(map50_95_cls, 4),
            }
        )

    # Print summary table to console
    df_metrics = pd.DataFrame(per_class_summary)
    print("\n" + "=" * 30 + " FINAL TEST SCORES " + "=" * 30)
    print(df_metrics.to_string(index=False))
    print("-" * 79)
    print(
        f"OVERALL MEAN -> Precision: {overall_p:.4f} | Recall: {overall_r:.4f} | F1: {overall_f1:.4f} | mAP@50: {overall_map50:.4f} | mAP@50-95: {overall_map50_95:.4f}"
    )
    print("=" * 79)

    # Export metrics to JSON and CSV
    final_scores_payload = {
        "overall": {
            "precision": overall_p,
            "recall": overall_r,
            "f1_score": overall_f1,
            "mAP50": overall_map50,
            "mAP50_95": overall_map50_95,
        },
        "per_class": per_class_summary,
    }

    json_path = os.path.join(args.output_dir, "final_test_scores.json")
    csv_path = os.path.join(args.output_dir, "final_test_scores.csv")

    with open(json_path, "w") as f:
        json.dump(final_scores_payload, f, indent=4)
    df_metrics.to_csv(csv_path, index=False)

    print(f"\n[Saved] Metrics JSON saved to: {json_path}")
    print(f"[Saved] Metrics CSV saved to:  {csv_path}")

    # 2. RUN INFERENCE & SAVE ANNOTATED IMAGES
    print("\n[Step 2/2] Generating and saving visual test predictions...")
    test_images_dir = os.path.join(
        os.path.dirname(args.data), "test", "images"
    )

    if os.path.exists(test_images_dir):
        model.predict(
            source=test_images_dir,
            conf=args.conf,
            iou=args.iou,
            save=True,  # Saves annotated visual images with boxes/labels
            save_txt=args.save_txt,  # Saves predicted bounding box coordinates
            project=args.output_dir,
            name="visual_predictions",
            exist_ok=True,
        )
        print(
            f"[Saved] Visual annotated test images saved to: {os.path.join(args.output_dir, 'visual_predictions')}"
        )
    else:
        print(
            f"Warning: Test images directory '{test_images_dir}' not found. Skipped image saving."
        )

    print("\n" + "=" * 70)
    print("Test Evaluation Complete!")
    print(f"All outputs stored inside: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()