# Week 4 – Keypoint Descriptors and Retrieval Normalization

This week focused on extending the CBIR pipeline with **local feature-based descriptors** and **unknown detection** to improve retrieval robustness.

---

## Folder Structure and Main Files

This week’s implementation includes several scripts and notebooks:

- **`week4_functions.py`**  
  Contains all the **new functions** developed for this week’s tasks.  
  It integrates and improves upon methods from previous weeks, handling:
  - Keypoint extraction and matching (SIFT, ORB, COLOR-SIFT)
  - Match filtering using Lowe’s ratio test
  - Score normalization by the number of detected keypoints
  - Threshold selection by maximizing the F1 score
  - Visualization utilities for query–database match inspection

- **Supporting `.py` scripts**  
  All other `.py` files (e.g., from previous weeks or groups) contain **legacy functions** reused in `week4_functions.py`.  
  These include utilities for noise removal, image splitting, etc.

- **`retrieval_system.ipynb`**  
  The **main notebook** for Week 4.  
  It runs all experiments and visualizations, including:
  - Retrieval performance evaluation (mAP@1, mAP@5)
  - Known vs. unknown query classification
  - Score normalization analysis
  - Plots and results shown in the final presentation slides

- **`images_for_presentation.ipynb`**  
  A **utility notebook** used to generate and save intermediate figures and visualizations for the project’s final presentation.  
  These include visual examples of keypoint detection, match visualization, and threshold effects.

- **`test.ipynb`**  
  The notebook used to **compute predictions for the test set** and generate the final **pickle submission file** for evaluation.

---

## Summary of the Week

The main objective of Week 4 was to complement the global color and texture descriptors from previous weeks with **local descriptors (SIFT and ORB)**, and to introduce a normalization strategy that accounts for variability in the number of detected features.

**Key outcomes:**
- Enhanced retrieval discriminability through **normalized matching scores**.  
- Reliable **threshold-based classification** between known and unknown queries.  
- Integration of all previous modules into a final, end-to-end CBIR pipeline.  

> For the complete workflow, see the `retrieval_system.ipynb` notebook and the implementation in `week4_functions.py`.

## Notes

Running the experiments will automatically create some cache folders for storing intermediate results.  
These are already included in `.gitignore` to avoid being pushed to GitHub.

## Authors

**Team 2 – MCV 2025 C1 Project**
