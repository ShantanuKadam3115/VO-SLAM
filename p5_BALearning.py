import cv2 as cv
import numpy as np

from vslam.config import data_root_path
from vslam.kitti_io import load_calib, load_poses, load_image
from vslam.features import detect, match_descriptors

seq = "00"
seq_dir = data_root_path / "sequences" / seq
gt = load_poses(data_root_path / f"poses/{seq}.txt")
K, B = load_calib(seq_dir / "calib.txt")
N = len(gt)


f = K[0][0]
Cx = K[0][2]
Cy = K[1][2]


img_K0_L = load_image(seq_dir, 0, 0)
img_K0_R = load_image(seq_dir, 0, 1)


# feature matching
orb = cv.ORB_create(nfeatures=3000)
bf = cv.BFMatcher(cv.NORM_HAMMING)

pts_K0_L, des_K0_L = detect(img_K0_L, orb)
pts_K0_R, des_K0_R = detect(img_K0_R, orb)

idx_L, idx_R = match_descriptors(des_K0_L, des_K0_R, bf)

pts1 = pts_K0_L[idx_L]
pts2 = pts_K0_R[idx_R]

# stereo triangulation.

U_K0_L = pts1[:,0]
V_K0_L = pts1[:,1]

U_K0_R = pts2[:,0]
V_K0_R = pts2[:,1]

initial_disparity = U_K0_L - U_K0_R

d_min = 1
d_max = (f*B)/ 2


row_mask = np.abs(V_K0_L - V_K0_R) < 2
range_mask = (initial_disparity < d_max) & (initial_disparity > d_min)
valid = row_mask & range_mask

disparity = initial_disparity[valid]
u = U_K0_L[valid]
v = V_K0_L[valid]

Z = (f*B)/disparity
X = (u - Cx) * Z / f
Y = (v - Cy) * Z / f

World_points = np.column_stack((X,Y,Z))

# descriptor of each 3D point = descriptor of its K0 left keypoint
World_des = des_K0_L[idx_L][valid]

# Matching at frame k1(temporal match)

img_K1_L = load_image(seq_dir, 1, 0)
# img_K1_R = load_image(seq_dir, 1, 1)

pts_K1_L, des_K1_L = detect(img_K1_L, orb)

idx_W, idx_K1 = match_descriptors(World_des, des_K1_L, bf)

matched_World_points = World_points[idx_W]
matched_pts_K1_L = pts_K1_L[idx_K1]


#ransac for getting R,t

ok, rvec, tvec, inl = cv.solvePnPRansac(matched_World_points, matched_pts_K1_L, K, None, iterationsCount=200, reprojectionError= 3.0, confidence=0.999)

T_wc = np.eye(4)
T_wc[:3, :3] = cv.Rodrigues(rvec)[0]
T_wc[:3, 3] = tvec.ravel()

#predict and compute the residuals
R = T_wc[:3, :3]
t = T_wc[:3, 3]

X_cam = R@ matched_World_points.mT + t.reshape(3, 1)

# print(R.shape, t.shape, matched_World_points.mT.shape)


