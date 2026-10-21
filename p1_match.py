import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from vslam.config import RESULTS_DIR, data_root_path
from vslam.features import match_features
from vslam.kitti_io import load_calib, load_image, load_poses
from vslam.metrics import rotation_angle_deg

seq_dir = data_root_path / "sequences/00"
calib_dir = seq_dir / "calib.txt"

## Consecutive frames from the same (left) camera
img1 = load_image(seq_dir, 0)
img2 = load_image(seq_dir, 1)

K, _ = load_calib(calib_dir)

## Create once, reuse for every frame pair.
## ORB descriptors are binary -> Hamming distance; no crossCheck with knnMatch(k=2).
orb = cv.ORB_create(nfeatures=3000)
bf = cv.BFMatcher(cv.NORM_HAMMING)


pts1, pts2 = match_features(img1, img2, orb, bf)
print(f"matches after ratio test: {len(pts1)}")

## 5-point RANSAC: E has 5 DoF (3 rotation + 2 for the direction of t; scale is unobservable).
## threshold = max pixel distance from a point to its epipolar line to count as an inlier.
E, mask = cv.findEssentialMat(pts1, pts2, K, method=cv.RANSAC, prob=0.999, threshold=1.0)
inliers = mask.ravel().astype(bool)
print(f"RANSAC inliers: {inliers.sum()} / {len(pts1)}")

## E decomposes into 4 (R, t) candidates; the cheirality check keeps the one that puts
## the points in front of both cameras. Convention: X_2 = R X_1 + t, with |t| = 1.
n_good, R_est, t_est, _ = cv.recoverPose(E, pts1, pts2, K, mask=mask.copy())
print(f"passed cheirality: {n_good}")


## Draw each match as an arrow from its frame-0 position to its frame-1 position, on frame 1.
## Inliers green, outliers red (BGR).
vis = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
for (x0, y0), (x1, y1), ok in zip(pts1, pts2, inliers):
    color = (0, 255, 0) if ok else (0, 0, 255)
    cv.arrowedLine(vis, (round(x0), round(y0)), (round(x1), round(y1)),
                   color, 1, tipLength=0.3)

save_path = RESULTS_DIR / "p1_matches_00_0-1.png"
save_path.parent.mkdir(parents=True, exist_ok=True)
cv.imwrite(str(save_path), vis)

## Compare with ground truth.
## KITTI T[i] maps camera-i coords -> frame-0 coords, so T_rel = inv(T[0]) @ T[1]
## maps frame 1 -> frame 0. OpenCV's (R, t) map frame 0 -> frame 1, so compare
## against inv(T_rel).
gt = load_poses(data_root_path / "poses/00.txt")
T_rel = np.linalg.inv(gt[0]) @ gt[1]
T_gt = np.linalg.inv(T_rel)
R_gt, t_gt = T_gt[:3, :3], T_gt[:3, 3]

rot_err = rotation_angle_deg(R_gt.T @ R_est)

## Scale is unknown, so only the direction of t can be compared
t_gt_hat = t_gt / np.linalg.norm(t_gt)
t_est_hat = t_est.ravel() / np.linalg.norm(t_est)
trans_dir_err = np.degrees(np.arccos(np.clip(t_gt_hat @ t_est_hat, -1.0, 1.0)))

print(f"GT motion:  {rotation_angle_deg(R_gt):.3f} deg, |t| = {np.linalg.norm(t_gt):.3f} m")
print(f"t_gt dir:   {np.round(t_gt_hat, 3)}")
print(f"t_est dir:  {np.round(t_est_hat, 3)}")
print(f"rotation error:        {rot_err:.3f} deg")
print(f"translation dir error: {trans_dir_err:.3f} deg")

plt.figure(figsize=(14, 5))
plt.imshow(cv.cvtColor(vis, cv.COLOR_BGR2RGB))
plt.title(f"ORB matches, frame 0 -> 1: {inliers.sum()} inliers (green), "
          f"{(~inliers).sum()} outliers (red)")
plt.axis("off")
plt.show()
