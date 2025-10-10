# Image Retrieval System - Week 2

This module implements a Content-Based Image Retrieval (CBIR) system using optimized algorithms developed in Week 1. It allows you to retrieve the most similar images from a database (BBDD) given a query image, using two best-performing methods based on color histograms and similarity measures.

## Features
- **Efficient retrieval** of similar images from a database using color histograms (CIELAB, HSV, HLS).
- **Two optimized retrieval methods** (method1 and method2) with different descriptor and similarity measure combinations.
- **Caching** of computed histograms for fast repeated queries.
- **Evaluation** of retrieval performance using MAP@K on the qsd1_w1 dataset.

## Usage

### 1. Requirements
- Python 3.9 or higher.
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### 2. Running the Example
From the `Week2` directory, run:
```bash
python image_retrieval.py
```
This will:
- Initialize the retrieval system and cache histograms if not already cached.
- Run example queries (see `main()` in the script).
- Print the top 10 most similar images for a sample query using both methods.
- Show method configuration details.
- Evaluate both methods on the qsd1_w1 dataset and print MAP@1 and MAP@5 scores.

### 3. Using the Retrieval System in Your Code
You can import and use the `ImageRetrieval` class in your own scripts:
```python
from image_retrieval import ImageRetrieval
retriever = ImageRetrieval(database_path="../BBDD/", cache_path="cache/")
results = retriever.retrieve_similar_images("path/to/query.jpg", method="method1", k=10)
print(results)
```

### 4. Methods
- **method1**: Uses CIELAB and HLS histograms with a specific combination of similarity measures and weights.
- **method2**: Uses CIELAB and HSV histograms with a different combination of similarity measures and weights.

You can get method details with:
```python
info = retriever.get_method_info("method1")
print(info)
```

### 5. Evaluation
To evaluate retrieval performance on the qsd1_w1 dataset:
```python
results = retriever.evaluate_on_qsd1(query_path="../qsd1_w1/", gt_path="../qsd1_w1/gt_corresps.pkl", k_values=[1, 5])
print(results)
```

## File Structure
- `image_retrieval.py`: Main retrieval system implementation.
- `BBDD/`: Image database folder.
- `cache/`: Stores cached histograms for fast access.
- `qsd1_w1/`, `qst1_w1/`: Query and ground truth folders for evaluation.

## Notes
- Make sure the database and query image paths are correct relative to your working directory.
- The first run may take longer due to histogram computation and caching.

## Authors
Team 2 - MCV 2025 C1 Project
