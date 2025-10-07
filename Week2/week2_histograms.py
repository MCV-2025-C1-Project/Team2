import cv2
import numpy as np

class Histogram2D:
    """
    Compute a 2D histogram for two color channels, e.g. H-S in HSV space.
    """
    def __init__(self, bins=(32, 32), color_space='HSV'):
        self.bins = bins
        self.color_space = color_space
        self.hist = None
        self.normalized = None

    def compute(self, image):
        if self.color_space == 'HSV':
            img_cs = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            channels = [0, 1]  # H and S
            ranges = [0, 180, 0, 256]
        elif self.color_space == 'LAB':
            img_cs = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
            channels = [1, 2]  # a and b
            ranges = [0, 256, 0, 256]
        else:
            raise ValueError("Unsupported color space for 2D histogram")

        hist = cv2.calcHist([img_cs], channels, None, self.bins, ranges)
        self.hist = cv2.normalize(hist, hist).flatten()
        self.normalized = self.hist
        return self.normalized


class Histogram3D:
    """
    Compute a 3D color histogram for an image (e.g. RGB or Lab).
    """
    def __init__(self, bins=(8, 8, 8), color_space='RGB'):
        self.bins = bins
        self.color_space = color_space
        self.hist = None
        self.normalized = None

    def compute(self, image):
        if self.color_space == 'RGB':
            img_cs = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ranges = [0, 256, 0, 256, 0, 256]
        elif self.color_space == 'LAB':
            img_cs = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
            ranges = [0, 256, 0, 256, 0, 256]
        else:
            raise ValueError("Unsupported color space for 3D histogram")

        hist = cv2.calcHist([img_cs], [0, 1, 2], None, self.bins, ranges)
        self.hist = cv2.normalize(hist, hist).flatten()
        self.normalized = self.hist
        return self.normalized


class BlockHistogram:
    """
    Divide image into blocks (grid_x × grid_y), compute histogram per block, and concatenate.
    """
    def __init__(self, bins=(8, 8, 8), grid=(2, 2), color_space='RGB'):
        self.bins = bins
        self.grid = grid
        self.color_space = color_space

    def compute(self, image):
        h, w, _ = image.shape
        block_h, block_w = h // self.grid[0], w // self.grid[1]
        features = []

        for i in range(self.grid[0]):
            for j in range(self.grid[1]):
                block = image[i * block_h:(i + 1) * block_h,
                              j * block_w:(j + 1) * block_w]
                hist = Histogram3D(self.bins, color_space=self.color_space).compute(block)
                features.append(hist)
        return np.concatenate(features)


class SpatialPyramidHistogram:
    """
    Compute hierarchical block histograms at multiple levels (spatial pyramid representation).
    """
    def __init__(self, bins=(8, 8, 8), levels=3, color_space='RGB'):
        self.bins = bins
        self.levels = levels
        self.color_space = color_space

    def compute(self, image):
        pyramid_features = []
        for l in range(self.levels):
            grid = (2 ** l, 2 ** l)
            hist = BlockHistogram(self.bins, grid, color_space=self.color_space).compute(image)
            pyramid_features.append(hist)
        return np.concatenate(pyramid_features)
