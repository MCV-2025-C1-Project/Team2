# C1 Project

Welcome to **Team 2’s project** for the **C1 course** from the **Master in Computer Vision (MCV 2025–26)**.

This project implements a **Content-Based Image Retrieval (CBIR)** system capable of retrieving information about artworks (or other visual objects) from a database given a query image.  
The system compares visual features using **histogram-based descriptors** and **similarity measures** to find the most relevant match.  

---

## Installation

This project was developed using **Python 3.9**, but any version **≥ 3.8** should work.

To clone and set up the repository:

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
├── .gitignore                # Files and folders excluded from Git
├── README.md                 # Main project description
├── requirements.txt          # Python dependencies
├── Week1/                    # Week 1 experiments and analysis
├── Week2/                    # Week 2 advanced tasks and improvements
├── Data/                     # Local datasets (not pushed to GitHub)
│     ├── BBDD                # Main database of reference images
│     ├── Week1
│     │     ├── qsd1_w1       # Query set 1 (development)
│     │     └── qst1_w1       # Query set 1 (test)
│     └── Week2
│     │   ├── qsd1_w2         # Query set 1 (development)
│     │   ├── qsd2_w2         # Query set 2 (development)
│     └── qst_w2              # Test data for week 2
│     │   ├── qst1_w2         # Test set 1
│     └── qst2_w2             # Test set 2
└── Results/                  # Output and submission files
```

**Note**: The Data/ directory contains all image datasets and must follow the exact structure above for the code to run correctly. Its datasets are not pushed to GitHub and should be stored locally.

## Week 1 – Baseline CBIR Pipeline

The first week focused on designing and analyzing **image descriptors** to build a baseline CBIR system.

We implemented multiple **color-space 1D histograms** (RGB, HSV, CIELAB, YCbCr, HLS, etc.) and **similarity measures** (Euclidean, L1, Chi-square, Correlation, etc.).  
Experiments were carried out to:

1. Evaluate all descriptor–similarity combinations.  
2. Combine the best descriptors with learned weights.  
3. Optimize the histogram bin sizes.

The resulting system effectively retrieves similar images from the database and establishes a solid baseline for later improvements.

> For detailed usage, refer to the **Week1/README.md** file inside the folder.

---

## Week 2 – Advanced Histogram Methods and Background Removal

The second week expanded the baseline with **new histogram descriptors** (2D, 3D, block-based, and pyramid histograms) and introduced a **background removal system** for better retrieval performance on artwork images.

The work includes:
- Evaluating and comparing new histogram structures.  
- Implementing an automatic background extraction method.  
- Combining both modules into a unified retrieval pipeline.  
- Generating final submission files for evaluation.

> Detailed instructions are provided in **Week2/README.md**.

---

## Week 3 – Texture Descriptors and Multi-painting Detection

This week introduced **texture-based image descriptors** and an improved **background removal pipeline** to handle images containing one or two paintings.

The main contributions were:
- Implementation of **Local Binary Patterns (LBP)** and **Discrete Cosine Transform (DCT)** descriptors, with analyses of:
  - Multi-scale vs. single-scale LBP  
  - Number of DCT coefficients  
  - Combination of LBP + DCT  
- Comparison of color spaces (RGB, HSV, CIELAB) for texture extraction.  
- Integration of **denoising preprocessing** to improve descriptor stability.  
- Development of an automatic **painting segmentation** module capable of detecting multiple artworks per query image.

These components were merged into a unified **retrieval pipeline** combining color, texture, and structure cues.

> For detailed usage and experiments, see **Week3/README.md**.

---

## Week 4 – Keypoint Descriptors and Unknown Detection

The final week focused on **local feature-based retrieval** using keypoint descriptors and methods to improve robustness.

Key elements included:
- Implementation and evaluation of **SIFT**, **ORB** and **COLOR-SIFT** descriptors for database-query matching.  
- Development of a **matching-score normalization** method based on the number of detected keypoints, improving discriminability between known and unknown queries.  
- Threshold optimization using **F1-score maximization** to classify queries as *known* or *unknown*.  
- Comprehensive visualization tools for keypoints, matches, and retrieval results.

These improvements completed the CBIR system by combining **global descriptors** (histograms, textures) and **local descriptors** (SIFT/ORB) for more accurate retrieval and recognition of artworks.

> For detailed implementation and results, see **Week4/** notebooks and scripts.

---

## Team Members

**Team 2 – MCV 2025 C1 Project**

- **Adrià Ruiz Puig**  
- **Pau Monserrat Llabrés**  
- **Souparni Mazumder**  
- **Benet Ramió Comas**
