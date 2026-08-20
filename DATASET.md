# Dataset Documentation

# Dataset Documentation

**Dataset Link:** [Click here to view/download the dataset](https://universe.roboflow.com/rj-xvfw4/pallet-ff5lh-fp7tw/dataset/1)

## 1. Sourcing & Cost
* **Where the data came from and why:** The dataset was sourced from [mention Roboflow Universe, Kaggle, or manual scraping]. Given the 5-day time constraint, leveraging an existing pre-annotated dataset for pallets was prioritized over manually scraping and labeling thousands of images from scratch.
* **The Cost of this Choice:** While it saved significant time, the tradeoff was a lack of control over warehouse-specific lighting conditions, camera angles, and the absence of a reliable `box` class. This forced the SOP compliance logic to be implemented mathematically rather than via an end-to-end vision model.

## 2. Dataset Metrics & Splits
* **Classes:** 2 classes (`pallet`, `hole`).
* **Annotation Counts:** [Insert total number of images and total number of bounding boxes here].
* **Split Protocol:** The dataset was split into Train/Valid/Test using a [e.g., 70/20/10] ratio. 
* **What it was split by:** The split was randomized across the dataset [or mention if you stratified by image orientation/lighting].

## 3. Labeling Guidelines (One-Page Summary)
* **`pallet`:** Draw a tight bounding box around the absolute visible outer edges of the pallet structure. Do not include the boxes or load sitting on top of the pallet.
* **`hole`:** Draw a bounding box around the internal negative space of the forklift entry points. If the hole is partially occluded by a fork or debris, bound only the visible opening.

## 4. Known Biases and Gaps
* **Angle Bias:** The dataset heavily features ground-level or eye-level photography, which differs from the required 1.2m height and 20-degree downward tilt. I attempted to correct this via rotational data augmentation.
* **Material Bias:** The dataset predominantly features standard wooden stringer pallets. Blue plastic pallets (like the one shown in our failure analysis) are underrepresented.
* **Occlusion Gap:** There is a lack of images showing heavy stretch-wrap glare obscuring the pallet structure, which is a common reality in live warehouses.
