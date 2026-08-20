import os
import json
import glob
from pose_estimator import PalletPoseEstimator

def process_yolo_labels(label_dir, img_w=640, img_h=640):
    """Parses YOLO txt files, associates holes to pallets, and runs pose estimation."""
    
    # Initialize the mathematical engine
    estimator = PalletPoseEstimator(cam_height=1.2, tilt_deg=20.0, img_w=img_w, img_h=img_h)
    
    # Setup for final JSON output (Section 4 requirement)
    final_report = {
        "assessment_meta": {
            "camera_height_m": 1.2,
            "camera_tilt_deg": 20.0,
            "usable_envelope": "2.0m to 6.0m"
        },
        "frames": []
    }
    
    # Find all .txt files from your test run
    txt_files = glob.glob(os.path.join(label_dir, "*.txt"))
    print(f"Found {len(txt_files)} label files to process...")

    for txt_file in txt_files:
        frame_id = os.path.basename(txt_file).replace('.txt', '.jpg')
        frame_data = {"frame_id": frame_id, "pallets": []}
        
        pallets = []
        holes = []

        # 1. Parse the YOLO text file
        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5: continue
                
                # YOLO format: class_id, x_center, y_center, width, height
                class_id = int(parts[0])
                u = float(parts[1]) * img_w
                v = float(parts[2]) * img_h
                w = float(parts[3]) * img_w
                h = float(parts[4]) * img_h
                
                det = {"center": (u, v), "bbox": (u - w/2, v - h/2, u + w/2, v + h/2)}
                
                if class_id == 1:  # Assuming 1 is 'pallet'
                    pallets.append(det)
                elif class_id == 0:  # Assuming 0 is 'hole'
                    holes.append(det)

        # 2. Association Logic: Match holes to pallets
        for idx, pallet in enumerate(pallets):
            p_xmin, p_ymin, p_xmax, p_ymax = pallet["bbox"]
            
            # Find holes that fall inside this pallet's bounding box
            assigned_holes = []
            for hole in holes:
                hx, hy = hole["center"]
                if p_xmin <= hx <= p_xmax and p_ymin <= hy <= p_ymax:
                    assigned_holes.append(hole["center"])
            
            pallet_output = {
                "pallet_id": f"{frame_id}_P{idx}",
                "holes_detected": len(assigned_holes),
                "pose": None,
                "status": "SUCCESS"
            }

            # 3. Pose Calculation Gatekeeper
            if len(assigned_holes) == 2:
                # Sort holes: left (smaller x) vs right (larger x)
                assigned_holes.sort(key=lambda pt: pt[0])
                hole_L, hole_R = assigned_holes[0], assigned_holes[1]
                
                # Run the math engine
                pos_x, pos_y, theta = estimator.estimate_pose(hole_L, hole_R)
                
                if pos_x is not None:
                    # Check if it falls outside the "usable envelope"
                    distance = (pos_x**2 + pos_y**2)**0.5
                    if distance > 6.0 or distance < 2.0:
                        pallet_output["status"] = "WARNING_OUT_OF_RANGE"
                    
                    pallet_output["pose"] = {
                        "x_meters": round(pos_x, 3),
                        "y_meters": round(pos_y, 3),
                        "orientation_deg": round(theta, 2)
                    }
                else:
                    pallet_output["status"] = "POSE_FAILED_HORIZON_ERROR"
                    
            elif len(assigned_holes) < 2:
                pallet_output["status"] = "POSE_FAILED_INSUFFICIENT_FEATURES"
            else:
                pallet_output["status"] = "POSE_FAILED_AMBIGUOUS_FEATURES" # 3+ holes found
                
            frame_data["pallets"].append(pallet_output)
            
        final_report["frames"].append(frame_data)

    # Export to JSON
    output_path = os.path.join(os.path.dirname(label_dir), "final_pose_report.json")
    with open(output_path, 'w') as f:
        json.dump(final_report, f, indent=4)
        
    print(f"\nProcessing complete! Report saved to: {output_path}")

if __name__ == "__main__":
    # Point this to the exact folder where your test.py saved the text files
    yolo_labels_folder = "/content/drive/MyDrive/Dhelivery_assingment/pallet.v1-pdd-1.yolov8/runs/test_evaluation/visual_predictions/labels"
    process_yolo_labels(yolo_labels_folder)