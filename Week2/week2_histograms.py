import cv2
import numpy as np


# -------- 2D HISTOGRAM -------- #
class Histogram2D:
    """
    Compute a 2D histogram for two color channels in any color space.
    Example:
        hist = Histogram2D((32, 32), 'HSV').compute(image)
    or
        hist = Histogram2D((32, 32), (0, 1), 'HSV').compute(image)

    Returns a normalized flattened histogram.
    """
    def __init__(self, bins=(32, 32), channels=(0, 1), color_space='HSV'):
        # allow calling Histogram2D((32,32), 'HSV')
        if isinstance(channels, str):
            color_space = channels
            channels = (0, 1)
        self.bins = tuple(int(b) for b in bins)
        self.channels = tuple(int(c) for c in channels)
        self.color_space = color_space.upper()
        self.hist = None

    def _convert(self, image):
        cs = self.color_space
        if cs == 'HSV':
            return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif cs == 'LAB':
            return cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        elif cs == 'HLS':
            return cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        elif cs == 'RGB':
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unsupported color space: {self.color_space}")

    def _ranges_for(self):
        if self.color_space == 'HSV':
            return [0, 180, 0, 256]
        return [0, 256, 0, 256]

    def compute(self, image, mask=None):
        img_cs = self._convert(image)
        ranges = self._ranges_for()
        hist = cv2.calcHist([img_cs], list(self.channels), mask, self.bins, ranges)
        hist = hist.astype('float32')
        if hist.sum() > 0:
            hist /= hist.sum()
        hist = hist.flatten()
        self.hist = hist
        return hist


# -------- 3D HISTOGRAM -------- #
class Histogram3D:
    """
    Compute a 3D color histogram for an image (RGB, HSV, LAB, HLS).
    Returns normalized flattened histogram.
    """
    def __init__(self, bins=(8, 8, 8), color_space='RGB'):
        self.bins = tuple(int(b) for b in bins)
        if len(self.bins) != 3:
            raise ValueError("bins must be a 3-tuple for 3D histograms")
        self.color_space = color_space.upper()
        self.hist = None

    def _convert(self, image):
        cs = self.color_space
        if cs == 'RGB':
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif cs == 'HSV':
            return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif cs == 'LAB':
            return cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        elif cs == 'HLS':
            return cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        else:
            raise ValueError(f"Unsupported color space: {self.color_space}")

    def _ranges_for(self):
        if self.color_space == 'HSV':
            return [0, 180, 0, 256, 0, 256]
        return [0, 256, 0, 256, 0, 256]

    def compute(self, image, mask=None):
        img_cs = self._convert(image)
        ranges = self._ranges_for()
        hist = cv2.calcHist([img_cs], [0, 1, 2], mask, self.bins, ranges)
        hist = hist.astype('float32')
        if hist.sum() > 0:
            hist /= hist.sum()
        hist = hist.flatten()
        self.hist = hist
        return hist


# -------- BLOCK HISTOGRAM -------- #
class BlockHistogram:
    """
    Divide image into grid_x × grid_y blocks, compute a 3D histogram per block, concatenate.
    Handles leftover pixels so all pixels are covered.
    """
    def __init__(self, bins=(8, 8, 8), grid=(2, 2), color_space='RGB'):
        self.bins = tuple(int(b) for b in bins)
        self.grid = tuple(int(g) for g in grid)
        self.color_space = color_space

    def compute(self, image, mask=None):
        h, w = image.shape[:2]
        gx, gy = self.grid
        features = []

        ys = [0] + [(i * h) // gx for i in range(1, gx)] + [h]
        xs = [0] + [(j * w) // gy for j in range(1, gy)] + [w]

        for i in range(gx):
            for j in range(gy):
                y0, y1 = ys[i], ys[i + 1]
                x0, x1 = xs[j], xs[j + 1]
                block = image[y0:y1, x0:x1]
                block_mask = None
                if mask is not None:
                    block_mask = mask[y0:y1, x0:x1]
                hist = Histogram3D(self.bins, color_space=self.color_space).compute(block, mask=block_mask)
                features.append(hist)

        return np.concatenate(features) if features else np.array([])


# -------- SPATIAL PYRAMID HISTOGRAM -------- #
class SpatialPyramidHistogram:
    """
    Hierarchical block histograms with optional level weighting.
    levels: number of levels (level 0 = whole image)
    weights: 'uniform', 'geometric', or custom list of per-level weights.
    """
    def __init__(self, bins=(8, 8, 8), levels=3, color_space='RGB', weights='uniform'):
        self.bins = tuple(int(b) for b in bins)
        self.levels = int(levels)
        self.color_space = color_space
        self.weights = weights

    def _weight_for_level(self, level):
        if isinstance(self.weights, (list, tuple, np.ndarray)):
            if level < len(self.weights):
                return float(self.weights[level])
            return 1.0
        if self.weights == 'geometric':
            return 1.0 / (2 ** level)
        return 1.0  # uniform

    def compute(self, image, mask=None):
        pyramid_features = []
        for level in range(self.levels):
            grid = (2 ** level, 2 ** level)
            bh = BlockHistogram(self.bins, grid, color_space=self.color_space)
            hist = bh.compute(image, mask=mask)
            hist *= self._weight_for_level(level)
            pyramid_features.append(hist)

        return np.concatenate(pyramid_features) if pyramid_features else np.array([])


# -------- QUICK TEST -------- #
if __name__ == "__main__":
    # simple gradient image to test histogram lengths
    img = np.zeros((100, 80, 3), dtype=np.uint8)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            img[y, x] = [x % 256, y % 256, (x + y) % 256]  # B, G, R gradient

    h2 = Histogram2D((32, 32), 'HSV').compute(img)
    h3 = Histogram3D((8, 8, 8), 'RGB').compute(img)
    bh = BlockHistogram((8, 8, 8), (2, 2), 'RGB').compute(img)
    sp = SpatialPyramidHistogram((8, 8, 8), levels=3, color_space='RGB').compute(img)

    print("2D HS hist length:", h2.shape)
    print("3D hist length:", h3.shape)
    print("Block hist length:", bh.shape)
    print("Pyramid hist length:", sp.shape)
