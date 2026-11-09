import cv2 as cv
import numpy as np

from vslam.config import data_root_path
from vslam.kitti_io import load_calib, load_poses, load_image
from vslam.features import match_features

seq = "00"
seq_dir = data_root_path / "sequences" / seq
gt = load_poses(data_root_path / f"poses/{seq}.txt")
K, B = load_calib(seq_dir / "calib.txt")
N = len(gt)

# print(K)

img_K0_L = load_image(seq_dir, 0, 0)
img_K0_R = load_image(seq_dir, 0, 1)

img_K1_L = load_image(seq_dir, 1, 0)
img_K1_L = load_image(seq_dir, 1, 1)

orb = cv.ORB_create(nfeatures=3000)
bf = cv.BFMatcher(cv.NORM_HAMMING)

pts1, pts2 = match_features(img_K0_L, img_K0_R, orb, bf)

print(pts1, pts2)