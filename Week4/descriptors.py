"""
Computes descriptors for all images in a folder.
If precomputed keypoints exist, they will be used; otherwise, detection is done automatically.

Usage:
    python descriptors.py --input Data/BBDD --kp results/keypoints/museum --output results/descriptors/BBDD --method SIFT
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


def load_keypoints(kp_path, filenames):
    keypoints_list = []
    for fname in filenames:
        base = os.path.splitext(fname)[0]
        kp_file = os.path.join(kp_path, f"{base}_kp.npy")
        if not os.path.exists(kp_file):
            keypoints_list.append(None)
            continue
        data = np.load(kp_file)
        kp = [cv2.KeyPoint(x=float(k[0]), y=float(k[1]), _size=k[2], _angle=k[3],
                           _response=k[4], _octave=int(k[5]), _class_id=int(k[6])) for k in data]
        keypoints_list.append(kp)
    return keypoints_list


def create_extractor(method):
    method = method.upper()
    if method == 'SIFT':
        return cv2.SIFT_create()
    elif method == 'ORB':
        return cv2.ORB_create(nfeatures=1000)
    elif method == 'SURF':
        return cv2.xfeatures2d.SURF_create(400)
    elif method == 'HARRIS' or method == 'FAST':
        # These don't have descriptors — use BRIEF or BRISK as compatible extractors
        return cv2.xfeatures2d.BriefDescriptorExtractor_create()
    else:
        raise ValueError(f"Unsupported method: {method}")


def compute_descriptors(images, keypoints_list, extractor):
    all_descriptors = []
    updated_keypoints = []
    for img, kp in tqdm(zip(images, keypoints_list), total=len(images)):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if kp is None:
            # If no precomputed keypoints, detect automatically
            kp, desc = extractor.detectAndCompute(gray, None)
        else:
            kp, desc = extractor.compute(gray, kp)
        updated_keypoints.append(kp)
        all_descriptors.append(desc)
    return updated_keypoints, all_descriptors


def save_descriptors(filenames, keypoints, descriptors, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for fname, kp, desc in zip(filenames, keypoints, descriptors):
        base = os.path.splitext(fname)[0]
        np.save(os.path.join(out_dir, f"{base}_desc.npy"), desc)
        np.save(os.path.join(out_dir, f"{base}_kp.npy"),
                np.array([(p.pt[0], p.pt[1], p.size, p.angle, p.response, p.octave, p.class_id) for p in kp]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to input images')
    parser.add_argument('--kp', default=None, help='Path to precomputed keypoints (optional)')
    parser.add_argument('--output', required=True, help='Path to save descriptors')
    parser.add_argument('--method', default='SIFT', help='Method: SIFT, ORB, SURF, HARRIS, FAST')
    args = parser.parse_args()

    images, filenames = load_images_from_folder(args.input)
    keypoints_list = load_keypoints(args.kp, filenames) if args.kp else [None] * len(images)
    extractor = create_extractor(args.method)

    print(f"Computing descriptors using {args.method}...")
    keypoints, descriptors = compute_descriptors(images, keypoints_list, extractor)
    save_descriptors(filenames, keypoints, descriptors, args.output)
    print("Descriptors saved successfully.")


if __name__ == "__main__":
    main()
