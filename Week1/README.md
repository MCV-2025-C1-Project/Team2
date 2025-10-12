# Week 1 - Experimental CBIR Pipeline

This week focuses on building and evaluating a **Content-Based Image Retrieval (CBIR)** system using different histogram-based descriptors and similarity measures.

## Structure Overview

- **`main.ipynb`** — Main experimental notebook containing:
  - Descriptor extraction  
  - Similarity computation  
  - Evaluation and visualization of results  

- **`histograms.py`** — Functions for computing color histograms in multiple color spaces (CIELAB, HLS, HSV, etc.).

- **`similarity_measures.py`** / **`similarity_measures_optimized.py`** — Implementations of similarity metrics such as Euclidean, L1, and Chi² distances.

- **`mapk.py`** — Custom implementation of the **MAP@K (Mean Average Precision at K)** metric used to evaluate retrieval performance.

- **`helper_functions_main.py`** — Utility functions supporting data loading, normalization, and visualization.

## Experimental Pipeline

The notebook `main.ipynb` runs the complete pipeline:
1. Extracts image descriptors from the dataset using different color-space histograms.  
2. Computes pairwise similarities using multiple distance metrics.  
3. Evaluates performance using MAP@1 and MAP@5.  
4. Searches for optimal combinations of descriptors and weighting strategies.

The experiments aim to identify the most discriminative descriptors and weighting methods for the retrieval task.


## Notes

During execution, new folders (e.g., `cache/`, `submission/`) are automatically generated to store intermediate and final outputs.  
These are already listed in `.gitignore` and are **not pushed** to GitHub.


## Authors

**Team 2 – MCV 2025 C1 Project**
