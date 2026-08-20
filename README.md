# Computer-Vision-Assignment-Pallet-Pose-Estimation-Load-Compliance

# Pallet Pose Estimation & Load Compliance
**Role:** AI/ML Engineer (Computer Vision) Assessment - Delhivery  
**Deliverables:** Detection Pipeline, 3D Pose Estimator, SOP Compliance Logic, JSON Dispute Format

## 1. Approach and Significant Decisions 
Given the strict 5-day time budget, this system was architected to prioritize geometric reliability and modularity over brute-force deep learning.

*   **Decision 1: 2D Deep Learning + 3D Geometric Projection**
    *   *Approach:* Instead of training a complex end-to-end 3D pose neural network (which requires a massive synthetic dataset), I trained a highly augmented YOLOv8 model to strictly localize 2D pallet/hole coordinates. I then built a custom mathematical camera projection model (Pinhole model + Ray intersection) to extract metric X, Y, and theta coordinates.
    *   *Cost:* The system is entirely dependent on accurate physical camera calibration parameters (Height: 1.2m, Tilt: 20 degrees). If the camera is bumped, the math degrades.
*   **Decision 2: Heavy Augmentation over Large Data Volume**
    *   *Approach:* The initial model overfit to warehouse lighting. I countered this by maximizing YOLO's HSV shifts (`hsv_s: 0.8`), adding 15-degree rotational augmentations, and enabling MixUp. 
    *   *Cost:* Increased training time and complexity, but it completely eliminated the overfitting and forced the model to learn geometry over textures.
*   **Decision 3: Scoping Out the "Box" Vision Class**
    *   *Approach:* I deliberately scoped the neural network to only detect `pallet` and `hole`. 
    *   *Cost:* I cannot run an end-to-end vision test for the SOP compliance. However, *what I gained* was a completed assignment within the 5-day window. I wrote the mathematical bounding-box logic for SOP compliance and tested it via simulation, leaving the box-annotation dataset update for a future production sprint.

## 2. Results Distributions
Instead of point estimates, error and accuracy are reported as distributions across the evaluation set:

*   **Detection Accuracy:**
    *   mAP@50: **96.6%**
    *   mAP@50-95: **68.1%** (Proving highly tight bounding box edges, critical for the +/- 2cm pose tolerance).
*   **Pose Estimation Error Distributions:**
  *<img width="1000" height="400" alt="pose_error_distributions" src="https://github.com/user-attachments/assets/a172f874-7c25-4d3e-a5a0-6b0c9a73fa7d" />*
    *   Based on our sensitivity analysis, a +/- 1-degree camera tilt error creates exponential depth translation errors beyond 6m. Therefore, the system is mathematically bounded by a **usable envelope of 2.0m - 6.0m**.

## 3. Failure Analysis (The 3 Worst Cases)
The system was designed to fail gracefully. Here are the three hardest failure modes observed during testing:

1.  **Case 1: The Horizon Error**
    * <img width="512" height="512" alt="case1_horizon_error" src="https://github.com/user-attachments/assets/1d5d1557-6823-4daa-9dda-3755d11e2690" />

    *   *Root Cause:* The pallet was stacked extremely high in the image plane, crossing the camera's mathematical horizon line. The ray-casting algorithm resulted in a positive Z-vector (pointing at the ceiling, not the floor).
    *   *Handling:* The system caught the math error and successfully output `"status": "POSE_FAILED_HORIZON_ERROR"`.
2.  **Case 2: Extreme Occlusion**
    * <img width="512" height="512" alt="case2_occlusion" src="https://github.com/user-attachments/assets/11b53026-e32f-484f-ad20-2b9b08a52717" />

    *   *Root Cause:* The structural holes of the pallet are completely missing or occluded due to the upward camera angle and the metal warehouse racking blocking the lower half. As a result, the vision model detected 0 holes (visible as a lack of bounding boxes).
    *   *Handling:* The pose matrix requires exactly two points to calculate orientation. The system reverted to `"status": "POSE_FAILED_INSUFFICIENT_FEATURES"`.
3.  **Case 3: Usable Envelope Violation**
    * <img width="512" height="512" alt="case3_out_of_range" src="https://github.com/user-attachments/assets/42df4d2a-ee37-461e-b79e-4668d7c57fd0" />

    *   *Root Cause:* The pallet was too close or too far, moving into regions where pixel-quantization noise breaks the +/- 2cm tolerance.
    *   *Handling:* Pose is calculated, but explicitly flagged as `"status": "WARNING_OUT_OF_RANGE"`.

## 4. What I Couldn't Finish & Why
**End-to-End SOP Load Compliance Vision.** 
I did not train a master vision model to detect `box`, `pallet`, and `hole` simultaneously. Sourcing, cleaning, and annotating a reliable dataset of thousands of warehouse boxes would have taken an estimated 15-20 hours of manual labor, guaranteeing a blown 5-day deadline. 

I chose to implement the SOP-PAL-03 compliance checks via a mathematical bounding-box algorithm (`sop_compliance.py`) using simulated inputs. This proves I can engineer the core logic, while scoping the project intelligently to deliver a working pose-estimator on time.

## 5. AI Tool Usage
*   **Tools Used:** LLMs were used to scaffold boilerplate Python code (argparse, file I/O, JSON generation) and to accelerate writing the intrinsic/extrinsic transformation matrix formulas.
*   **What It Got Wrong:** During the testing phase, the AI generated a YOLO inference script with `exist_ok=True` that appended new predictions to old `.txt` files rather than overwriting them. This resulted in duplicate bounding boxes (e.g., 4 holes detected in a single pallet) which crashed the downstream pose logic with an `AMBIGUOUS_FEATURES` error. I identified the root cause, wiped the directory, and updated the pipeline to enforce clean-slate processing per run.

## 6. Deployment & Robustness (Target: Jetson Orin Nano)
*   **Latency & Export:** The YOLOv8m model was exported to ONNX format (included in the training script) for TensorRT deployment. On a 15W Jetson Orin Nano, this fp16 optimized engine is expected to easily exceed the target >= 15 FPS.
*   **Quantization Cost:** While INT8 quantization would maximize FPS, it often slightly degrades bounding box edge precision. Because our mathematical pose is highly sensitive to pixel-perfect bounding boxes to maintain the +/- 2cm tolerance, FP16 is the recommended deployment format.
*   **Temporal Smoothing:** Because pallets remain stationary for long periods, a deployed system should implement an Exponential Moving Average (EMA) on the X, Y, and orientation coordinates across multiple frames to eliminate single-frame pixel jitter.

## 7. SOP-PAL-03 Verification Triage
Assessing whether a single, downward-tilted side camera can verify the 8 warehouse load standards:

1.  **Overhang (>3cm):** *Verifiable.* Geometrically checked via bounding box X-axis boundaries.
2.  **Height < 1.8m:** *Partially Verifiable.* Assuming the top box sits directly over the center, we can project its 3D height. Fails if the pallet clips the top of the camera frame.
3.  **Aligned Columns / No Rotation:** *Verifiable.* Bounding box centroids and aspect ratios can flag rotational deviations.
4.  **Size Inversion (Heavy on top):** *Verifiable.* Checked by sorting bounding box areas by Y-axis height.
5.  **Stretch-Wrapped:** *Partially Verifiable.* Requires training a highly specialized model for surface glare, which is incredibly sensitive to warehouse lighting changes.
6.  **No Damaged Boxes:** *Not Verifiable.* The camera cannot see crushed boxes on the rear face of the pallet.
7.  **Centroid Alignment:** *Verifiable.* Calculated via the global center of mass of the detected box array.
8.  **Pallet Undamaged:** *Not Verifiable.* Split stringers under the load or on the rear face are completely occluded.
