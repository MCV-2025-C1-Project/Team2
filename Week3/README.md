# Week 2 - Usage

This week’s tasks are organized into four main groups, each corresponding to a primary file:

- **Task 1 & 2:** New histogram types and methods  
  _Main file:_ `new_histogram_types.ipynb`  
- **Task 3 & 4:** Background removal system  
  _Main file:_ `background_remover.ipynb`  
- **Task 5:** Complete pipeline (background removal + retrieval method)  
  _Main file:_ `task5_evaluation.py`  
- **Task 6:** Submission files  
  _Main file:_ `task6.py`


## New Histogram Types

The notebook includes all the experiments described in the slides to evaluate the best histogram descriptors using the new types (2D, 3D, block, and pyramid).  
Supporting functions are located in:  
`mapk.py`, `similarity_measures_optimized.py`, `week2_histograms.py`, and `new_histogram_helper_functions.py`.

- Run all the notebook cells to reproduce the results.  
- The **first execution** may take a long time since it computes the descriptors for the entire BBDD.  
  These results are **cached** for faster future runs.  
- The notebook also generates the **QST1-W2 submission** and the **visualizations** used in the slides.


## Background Removal System

Given an artwork image, this module generates a mask to remove its background.

Run `background_remover.ipynb` to see all the processing steps.  
Core functions are implemented in `background_remover.py`.

Additionally, `visualizer.ipynb` provides an **interactive web interface** to visualize and adjust the background removal process in real time.  


## Complete Pipeline

`task5_evaluation.py` combines the background removal system with the image retrieval method to handle images containing backgrounds.  
Running this script produces the evaluation metrics for the **development test**.


## Result Submission

`task6.py` contains the functions required to:
- Create the **submission masks**.  
- Generate the **pickle file** for **QST2-W2**.


## Notes

Running the experiments will automatically create some cache folders for storing intermediate results.  
These are already included in `.gitignore` to avoid being pushed to GitHub.


## Authors

**Team 2 – MCV 2025 C1 Project**
