import numpy as np
import json

def check_overhang(pallet_box, boxes):
    """Checks if any box extends beyond the pallet's X-axis boundaries."""
    px_min, py_min, px_max, py_max = pallet_box
    violations = []
    
    for i, box in enumerate(boxes):
        bx_min, by_min, bx_max, by_max = box["bbox"]
        # Allow a tiny 2% pixel tolerance for bounding box noise
        tolerance = (px_max - px_min) * 0.02 
        
        if bx_min < (px_min - tolerance) or bx_max > (px_max + tolerance):
            violations.append(f"Box {i} overhangs pallet edge.")
            
    return len(violations) == 0, violations

def check_alignment(boxes):
    """Checks if boxes form neat vertical columns based on center X coordinates."""
    if not boxes: return True, []
    
    violations = []
    centers_x = [(b["bbox"][0] + b["bbox"][2]) / 2 for b in boxes]
    
    mean_x = np.mean(centers_x)
    for i, cx in enumerate(centers_x):
        # If a box center deviates by more than 10% of the average box width, it's misaligned
        avg_width = np.mean([b["bbox"][2] - b["bbox"][0] for b in boxes])
        if abs(cx - mean_x) > (avg_width * 0.10):
            violations.append(f"Box {i} is misaligned from the column center.")
            
    return len(violations) == 0, violations

def check_size_sorting(boxes):
    """Checks if smaller boxes are placed on top of larger boxes."""
    if len(boxes) < 2: return True, []
    
    violations = []
    
    # Sort boxes by their Y-center (Y increases downwards in images, so largest Y is the bottom)
    boxes_sorted_by_height = sorted(boxes, key=lambda b: (b["bbox"][1] + b["bbox"][3]) / 2, reverse=True)
    
    for i in range(len(boxes_sorted_by_height) - 1):
        lower_box = boxes_sorted_by_height[i]
        upper_box = boxes_sorted_by_height[i + 1]
        
        lower_area = (lower_box["bbox"][2] - lower_box["bbox"][0]) * (lower_box["bbox"][3] - lower_box["bbox"][1])
        upper_area = (upper_box["bbox"][2] - upper_box["bbox"][0]) * (upper_box["bbox"][3] - upper_box["bbox"][1])
        
        # If the upper box is significantly larger than the lower box (20% tolerance)
        if upper_area > (lower_area * 1.2): 
            violations.append(f"Heavy/Large box found on top of smaller box.")
            break
            
    return len(violations) == 0, violations

def evaluate_pallet_sop(pallet_box, boxes):
    """Runs all SOP checks for a single pallet."""
    overhang_pass, overhang_msgs = check_overhang(pallet_box, boxes)
    align_pass, align_msgs = check_alignment(boxes)
    size_pass, size_msgs = check_size_sorting(boxes)
    
    return {
        "status": "PASS" if (overhang_pass and align_pass and size_pass) else "FAIL",
        "violations": overhang_msgs + align_msgs + size_msgs
    }

def run_simulation():
    print("="*60)
    print("SECTION 3: SOP LOAD COMPLIANCE ALGORITHM (SIMULATION)")
    print("="*60)
    
    # A hypothetical pallet bounding box [xmin, ymin, xmax, ymax]
    real_pallet = [200, 500, 400, 550] 
    
    print("\nTest Case 1: Perfect Pallet")
    perfect_boxes = [
        {"bbox": [220, 400, 380, 490]},  # Bottom layer (Large)
        {"bbox": [230, 300, 370, 390]},  # Middle layer (Medium)
        {"bbox": [250, 200, 350, 290]}   # Top layer (Small)
    ]
    result1 = evaluate_pallet_sop(real_pallet, perfect_boxes)
    print(json.dumps(result1, indent=2))
    
    print("\nTest Case 2: Non-Compliant Pallet (Overhang & Size Violation)")
    bad_boxes = [
        {"bbox": [250, 400, 350, 490]},  # Bottom layer (Small)
        {"bbox": [180, 300, 420, 390]},  # Middle layer (Massive - Overhangs pallet X=200/400)
    ]
    result2 = evaluate_pallet_sop(real_pallet, bad_boxes)
    print(json.dumps(result2, indent=2))

if __name__ == "__main__":
    run_simulation()