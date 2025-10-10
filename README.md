# C1 Project
Welcome to Team2's project for the C1 course from the MCV 25-26.

With this project, given a certain database of images (i.e. art from a museum) you can retrieve information of each element of the database with another image taken by you. This technique is called Content Based Image Retrieval (CBIR) and uses various histogram descriptors and similarity measures to find the right match and provide the relevant information. (Be aware that the algorithm is not perfect and it can fail and retrieve a different image by mistake.)

## Organization
In the next section you will find how to install and run the program. Beyond that point, we will organize the project by weeks, so we will explain what is added each week, give some insight on our decision for some implementations and give some final results we got to evaluate our algorithm.

## Installation
This project has been developed using Python 3.9, but any Python version above Python 3.8 should work.

To clone this repo you can do:

```bash
git clone https://github.com/MCV-2025-C1-Project/Team2.git && cd Team2
```

We recommend installing the required libraries inside a virtual environment:

**Option 1: Using Conda (Recommended)**

Create and activate the virtual environment:
```bash
conda create -n team2_env python=3.9
conda activate team2_env
```

**Option 2: Using venv (Alternative)**

Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Linux or macOS:
```bash
python -m venv .venv
source .venv/bin/activate
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

And you are good to go!

## Usage

The main implementation is in a Jupyter notebook which provides an interactive experience with visualizations. To run the notebook, open the main.ipynb file located in the Week1 folder and then simply run all cells in the notebook.

```
The system will:
1. Load or compute histograms (cached for faster re-runs)
2. Run optimization experiments to find the best algorithms
3. Generate test results for the qst1_w1 dataset
4. Create visualizations showing query results

```

## Repository Structure
```
Team2/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── BBDD/                       # Training database images (local only)
├── qsd1_w1/                    # Query set 1 (development) (local only)
├── qst1_w1/                    # Query set 1 (test) (local only)
└── Week1/                      # Main implementation
    ├── main.ipynb              # Main Jupyter notebook
    ├── histograms.py           # Histogram descriptor classes
    ├── similarity_measures_optimized.py  # Similarity measure functions
    ├── mapk.py                 # Evaluation metrics
    └── helper_functions_main.py # Utility functions
```

**Note**: The `BBDD/`, `qsd1_w1/`, and `qst1_w1/` directories contain the image datasets and are kept in local repositories only (not pushed to GitHub due to file size constraints).

## Week 1
During the first week, our objective was to design comprehensive image descriptors and explore different configurations to determine the most effective approach for image similarity matching. We implemented 9 different histogram descriptors (RGB, HSV, CIELAB, YCbCr, XYZ, HLS, YUV, Grayscale) and 9 similarity measures (Euclidean, L1, Chi-square, Histogram Intersection, Hellinger, Cosine, Bhattacharyya, Correlation, KL Divergence).

Our approach follows a systematic 3-stage optimization process:

1. **Experiment 1**: Individual performance analysis - testing all 81 combinations of descriptors and similarity measures
2. **Experiment 2**: Descriptor combination - finding optimal weighted combinations of descriptor pairs
3. **Experiment 3**: Bin count optimization - testing different histogram bin counts (8, 16, 32, 64, 128, 256)


The final system achieves strong performance with optimized HSV + CIELAB descriptor combinations and provides sub-second query processing after optimization.

## Team Members
This is the awesome team who collaborated in this project:

- **Adrià Ruiz Puig**
- **Pau Monserrat Llabrés**  
- **Souparni Mazumder**
- **Benet Ramió Comas**
