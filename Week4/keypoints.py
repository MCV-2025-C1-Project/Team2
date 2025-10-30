"""
Detects keypoints in all images from a given folder and saves them as .npy files.

Usage:
    python keypoints.py --input Data/BBDD --output results/keypoints/BBDD --method HARRIS
"""

import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm


def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
                filenames.append(filename)
    return images, filenames


def detect_keypoints(img, method='SIFT'):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    method = method.upper()

    if method == 'HARRIS':
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)
        kp = np.argwhere(dst > 0.01 * dst.max())
        keypoints = [cv2.KeyPoint(float(x[1]), float(x[0]), 3) for x in kp]
    elif method == 'FAST':
        fast = cv2.FastFeatureDetector_create()
        keypoints = fast.detect(gray, None)
    elif method == 'SIFT':
        detector = cv2.SIFT_create()
        keypoints = detector.detect(gray, None)
    elif method == 'ORB':
        detector = cv2.ORB_create()
        keypoints = detector.detect(gray, None)
    elif method == 'SURF':
        detector = cv2.xfeatures2d.SURF_create(400)
        keypoints = detector.detect(gray, None)
    else:
        raise ValueError(f"Unsupported method: {method}")

    return keypoints


def save_keypoints(filenames, keypoints, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for fname, kp in zip(filenames, keypoints):
        base = os.path.splitext(fname)[0]
        data = np.array([(p.pt[0], p.pt[1], p.size, p.angle, p.response, p.octave, p.class_id) for p in kp])
        np.save(os.path.join(out_dir, f"{base}_kp.npy"), data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to input image folder')
    parser.add_argument('--output', required=True, help='Path to output folder')
    parser.add_argument('--method', default='SIFT', help='Keypoint detector: HARRIS, FAST, SIFT, ORB, SURF')
    args = parser.parse_args()

    images, filenames = load_images_from_folder(args.input)
    all_keypoints = []

    print(f"Detecting keypoints using {args.method}...")
    for img in tqdm(images):
        kp = detect_keypoints(img, args.method)
        all_keypoints.append(kp)

    save_keypoints(filenames, all_keypoints, args.output)
    print("Keypoints saved successfully.")


if __name__ == "__main__":
    main()
