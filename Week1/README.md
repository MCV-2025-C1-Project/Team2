# Week1 - Experimental CBIR Pipeline

This folder contains the experimental pipeline for Content-Based Image Retrieval (CBIR) developed in Week1. The code here explores various histogram-based descriptors and similarity measures for image retrieval tasks.

## Structure
- `main.ipynb`: Jupyter notebook with the main experimental pipeline, including descriptor extraction, similarity computation, and evaluation.
- `histograms.py`: Functions for extracting color histograms in different color spaces (CIELAB, HLS, HSV, etc.).
- `similarity_measures.py` / `similarity_measures_optimized.py`: Functions for computing similarity between histograms (Euclidean, L1, Chi^2, etc.).
- `mapk.py`: Custom implementation of MAP@K (Mean Average Precision at K) metrics for evaluation.
- `helper_functions_main.py`: Helper functions for the pipeline.
- `cache/`: Stores cached descriptors for faster experimentation.
- `submission/`: Contains submission files and results.

Cache and submission folders are created during execution and are not included in the repository.