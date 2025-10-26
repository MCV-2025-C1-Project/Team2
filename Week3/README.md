# Week 2 - Usage

This week’s tasks are organized into four main groups, each corresponding to a primary file:

- **Task 1:** Noise filtering  
  _Main file:_ `noise_filter.ipynb`
- **Task 2:** Implementation of texture histograms  
  _Main file:_ `descriptors.ipynb`  
- **Task 3:** Painting detection and background removal  
  _Main file:_ `background_remover.ipynb`  
- **Task 4:** Complete pipeline (background removal + retrieval method)  
  _Main file:_ `task4_evaluation.py`
- **Task 5:** Result submission  
  _Main file:_ `task5_evaluation.py`

## Noise filtering

The notebook includes all the experiments described in the slides to evaluate and find the best noise filtering method. Please, run all the notebook cells to reproduce the results.  
Supporting functions are located in: `noise_filter.py`.

## Implementation of texture histograms

This task implements and evaluates several **texture descriptors** for image retrieval, including:

- **Local Binary Patterns (LBP):**  
  - Single-scale and multi-scale histograms  
  - Block-based grids (configurable `grid_x × grid_y`)  
- **Discrete Cosine Transform (DCT):**  
  - Configurable number of coefficients (best performance found with `128` coefficients)  
 

**Experiments included:**

1. **Descriptor comparison:** Evaluate retrieval performance (mAP@1 and mAP@5) for LBP, DCT, and their combinations.  
2. **LBP scale:** Evaluate based on single vs multiscale  
3. **DCT coefficient analysis:** Evaluate different numbers of DCT coefficients.  
4. **Noise removal effect:** Compare descriptor performance with and without denoising (`preprocess_image`) applied to query and database images.  
5. **Color space analysis:** Evaluate the effect of converting images to HSV or LAB before extracting descriptors.  

**Functions and scripts:**  
- `descriptors.py` contains all core functions for descriptor extraction, combination, and evaluation.  
- `extract_descriptors()` allows selecting descriptors (`lbp`, `dct`, `lbp+dct`, `all`) and applying optional noise removal.  
- `compute_map_at_k()` computes mAP for retrieval evaluation.  
- All experiments automatically save results in `results/` for reproducibility.
- Run `experiments_task2_full.ipynb` to see all the experiments that are implemented.

## Painting detection and background removal

Given an image with at most two artworks, detect them and give the mask to remove the background.

Run `background_remover.ipynb` to see all the processing steps.  
Core functions are implemented in `background_remover.py`.

## Complete Pipeline

`task4_evaluation.py` combines the background removal system with the image retrieval method to handle images with at most two artworks per image, containing backgrounds.  
Running this script produces the evaluation metrics for the **development test**.

## Result Submission

`task5.py` contains the functions required to:
- Create the **submission masks**.  
- Generate the **pickle file** for **QST2-W2**.

## Notes

Running the experiments will automatically create some cache folders for storing intermediate results.  
These are already included in `.gitignore` to avoid being pushed to GitHub.

## Authors

**Team 2 – MCV 2025 C1 Project**
