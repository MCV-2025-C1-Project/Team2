import os
import matplotlib.pyplot as plt
import numpy as np


"""Stores and operates the mean RGB histogram (mean over channels)"""
class RGB_Mean_Histogram:
    range = 256

    def __init__(self, height, width):
        self.pixelNumber = width * height
        self.red = [0] * RGB_Mean_Histogram.range
        self.green = [0] * RGB_Mean_Histogram.range
        self.blue = [0] * RGB_Mean_Histogram.range
        self.mean = [0] * RGB_Mean_Histogram.range
        self.normalized = [0] * RGB_Mean_Histogram.range

    def setHist(self, color, hist):
        if color == "red":
            self.red = hist
        elif color == "green":
            self.green = hist
        elif color == "blue":
            self.blue = hist

    def calculate_1D_hist(self):
        for i in range(256):
            self.mean[i] = (self.red[i] + self.green[i] + self.blue[i]) // 3

    def normalize(self):
        for i in range(len(self.normalized)):
            self.normalized[i] = self.mean[i] / self.pixelNumber

    def show(self):
        plt.figure(figsize=(10, 5))
        x = range(RGB_Mean_Histogram.range)
        plt.plot(x, self.mean, color='black', label='Mean Histogram')
        plt.title('Mean 1D RGB Histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.show()


class RGB_Concat_Histogram:
    """Stores concatenated RGB histograms (all channels)"""
    range = 256

    def __init__(self, height, width):
        self.pixelNumber = width * height
        self.red = [0] * RGB_Concat_Histogram.range
        self.green = [0] * RGB_Concat_Histogram.range
        self.blue = [0] * RGB_Concat_Histogram.range
        self.concat = [0] * (3 * RGB_Concat_Histogram.range)
        self.normalized = [0] * (3 * RGB_Concat_Histogram.range)

    def setHist(self, color, hist):
        if color == "red":
            self.red = hist
        elif color == "green":
            self.green = hist
        elif color == "blue":
            self.blue = hist

    def calculate_concat_hist(self):
        self.concat = np.concatenate([self.red, self.green, self.blue])

    def normalize(self):
        self.normalized = np.array(self.concat) / self.pixelNumber

    def show(self):
        plt.figure(figsize=(10, 5))
        x = range(len(self.concat))
        plt.plot(x, self.concat, color='black', label='RGB Concat Histogram')
        plt.title('Concatenated RGB Histogram')
        plt.xlabel('Bin')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.show()


class HSV_Concat_Histogram:
    """Stores concatenated HSV histograms"""
    range = 256

    def __init__(self, height, width):
        self.pixelNumber = width * height
        self.h = [0] * HSV_Concat_Histogram.range
        self.s = [0] * HSV_Concat_Histogram.range
        self.v = [0] * HSV_Concat_Histogram.range
        self.concat = [0] * (3 * HSV_Concat_Histogram.range)
        self.normalized = [0] * (3 * HSV_Concat_Histogram.range)

    def setHist(self, h, s, v):
        self.h = h
        self.s = s
        self.v = v

    def calculate_concat_hist(self):
        self.concat = np.concatenate([self.h, self.s, self.v])

    def normalize(self):
        self.normalized = np.array(self.concat) / self.pixelNumber

    def show(self):
        plt.figure(figsize=(10, 5))
        x = range(len(self.concat))
        plt.plot(x, self.concat, color='black', label='HSV Concat Histogram')
        plt.title('Concatenated HSV Histogram')
        plt.xlabel('Bin')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.show()


class CIELAB_Concat_Histogram:
    """Stores concatenated CIELAB histograms"""
    range = 256

    def __init__(self, height, width):
        self.pixelNumber = width * height
        self.l = [0] * CIELAB_Concat_Histogram.range
        self.a = [0] * CIELAB_Concat_Histogram.range
        self.b = [0] * CIELAB_Concat_Histogram.range
        self.concat = [0] * (3 * CIELAB_Concat_Histogram.range)
        self.normalized = [0] * (3 * CIELAB_Concat_Histogram.range)

    def setHist(self, l, a, b):
        self.l = l
        self.a = a
        self.b = b

    def calculate_concat_hist(self):
        self.concat = np.concatenate([self.l, self.a, self.b])

    def normalize(self):
        self.normalized = np.array(self.concat) / self.pixelNumber

    def show(self):
        plt.figure(figsize=(10, 5))
        x = range(len(self.concat))
        plt.plot(x, self.concat, color='black', label='CIELAB Concat Histogram')
        plt.title('Concatenated CIELAB Histogram')
        plt.xlabel('Bin')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.show()


class YCbCr_Concat_Histogram:
    """Stores concatenated YCbCr histograms"""
    range = 256

    def __init__(self, height, width):
        self.pixelNumber = width * height
        self.y = [0] * YCbCr_Concat_Histogram.range
        self.cb = [0] * YCbCr_Concat_Histogram.range
        self.cr = [0] * YCbCr_Concat_Histogram.range
        self.concat = [0] * (3 * YCbCr_Concat_Histogram.range)
        self.normalized = [0] * (3 * YCbCr_Concat_Histogram.range)

    def setHist(self, y, cb, cr):
        self.y = y
        self.cb = cb
        self.cr = cr

    def calculate_concat_hist(self):
        self.concat = np.concatenate([self.y, self.cb, self.cr])

    def normalize(self):
        self.normalized = np.array(self.concat) / self.pixelNumber

    def show(self):
        plt.figure(figsize=(10, 5))
        x = range(len(self.concat))
        plt.plot(x, self.concat, color='black', label='YCbCr Concat Histogram')
        plt.title('Concatenated YCbCr Histogram')
        plt.xlabel('Bin')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.show()


class Gray_Histogram:
    """Class that stores and operates the gray histogram"""
    # Range of possible values for each pixel (from 0 to 255 -> 256 different values)
    range = 256
    
    def __init__(self, width, height):
        # Calculate the total size of the image (number of pixels)
        self.pixelNumber = width * height
        # Vector to store the total number of pixels for each gray level
        self.gray = [0] * Gray_Histogram.range
        # Vector to store the 1D normalized gray histogram
        self.normalized = [0] * Gray_Histogram.range
        
    # Set the histogram
    def setHist(self, hist):
        self.gray = hist
        
    # Calculate the normalized histogram
    def normalize(self):
        for i in range(len(self.normalized)):
            self.normalized[i] = self.gray[i] / self.pixelNumber
            
    def show(self):
        """Display the grayscale histogram"""
        plt.figure(figsize=(10, 5))
        x = range(Gray_Histogram.range)
        plt.plot(x, self.gray, color='black', label='Grayscale Histogram')
        plt.title('1D Grayscale Histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.show()