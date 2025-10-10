# hyperparameter_search.py

from image_retrieval import ImageRetrieval
from mapk import mapk

import itertools

class HyperparameterSearch:
    def __init__(self, retriever, k_values=[5]):
        self.retriever = retriever
        self.k_values = k_values
        self.results = {}

    def run_experiments(self, experiment_configs, query_path, gt_path):
        for name, config in experiment_configs.items():
            print(f"\n=== Running {name} ===")
            self.retriever.temp_config = config
            result = self.retriever.evaluate_on_qsd1(query_path, gt_path, self.k_values)
            self.results[name] = result
        return self.results


# ---------- EXPERIMENT CONFIGURATIONS ----------

def make_2d_experiments():
    color_spaces = ["HSV", "LAB", "HLS"]
    channel_pairs = [(0,1), (0,2), (1,2)]
    bins_list = [16, 32, 64]
    experiments = {}

    for cs in color_spaces:
        for ch_pair in channel_pairs:
            for bins in bins_list:
                name = f"2D_{cs}_{ch_pair}_bins{bins}"
                experiments[name] = {
                    "descriptors": [f"2D_{cs}_{ch_pair[0]}{ch_pair[1]}"],
                    "weights1": [1.0],
                    "weights2": [0.0],
                    "bins": bins,
                    "similarity_indices": [1]  # L1 distance
                }
    return experiments


def make_3d_experiments():
    color_spaces = ["RGB", "LAB", "HSV", "HLS"]
    bins_list = [8, 16, 32]
    experiments = {}
    for cs, bins in itertools.product(color_spaces, bins_list):
        name = f"3D_{cs}_{bins}bins"
        experiments[name] = {
            "descriptors": [f"3D_{cs}"],
            "weights1": [1.0],
            "weights2": [0.0],
            "bins": bins,
            "similarity_indices": [1]
        }
    return experiments


def make_block_experiments():
    color_spaces = ["RGB", "LAB"]
    grids = [(2,2), (3,3)]
    bins_list = [8, 16]
    experiments = {}
    for cs, grid, bins in itertools.product(color_spaces, grids, bins_list):
        name = f"BLOCK_{cs}_{grid[0]}x{grid[1]}_{bins}bins"
        experiments[name] = {
            "descriptors": [f"BLOCK_{cs}"],
            "weights1": [1.0],
            "weights2": [0.0],
            "bins": bins,
            "similarity_indices": [1],
            "grid": grid
        }
    return experiments


def make_pyramid_experiments():
    weights = {
        "uniform": [1, 1, 1],
        "geometric": [1, 0.5, 0.25]
    }
    bins_list = [8, 16]
    experiments = {}
    for wname, wvals in weights.items():
        for bins in bins_list:
            name = f"PYRAMID_RGB_{bins}bins_{wname}"
            experiments[name] = {
                "descriptors": ["3D_RGB_PYRAMID"],
                "weights1": [1.0],
                "weights2": [0.0],
                "bins": bins,
                "similarity_indices": [1],
                "level_weights": wvals
            }
    return experiments


# ---------- MAIN EXECUTION ----------

def main():
    retriever = ImageRetrieval()
    search = HyperparameterSearch(retriever)

    query_path = "../Data/Week1/qsd1_w1/"
    gt_path = "../Data/Week1/qsd1_w1/gt_corresps.pkl"

    # Combine all experiment sets
    all_experiments = {}
    all_experiments.update(make_2d_experiments())
    all_experiments.update(make_3d_experiments())
    all_experiments.update(make_block_experiments())
    all_experiments.update(make_pyramid_experiments())

    print(f"\nRunning {len(all_experiments)} experiments total...")

    results = search.run_experiments(all_experiments, query_path, gt_path)

    # Sort and show top results
    print("\n=== TOP 5 EXPERIMENTS BY MAP@5 ===")
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1]["method1"]["map_scores"]["MAP@5"],
        reverse=True
    )[:5]

    for name, res in sorted_results:
        score = res["method1"]["map_scores"]["MAP@5"]
        print(f"{name}: MAP@5 = {score:.4f}")


if __name__ == "__main__":
    main()
