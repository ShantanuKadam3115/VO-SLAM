import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from vslam.config import RESULTS_DIR, data_root_path
from vslam.features import match_features
from vslam.geometry import project, triangulate
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


## ---- Triangulation (DLT) of the RANSAC inliers ----

## Camera 1 is the world frame: P1 = K [I | 0], P2 = K [R | t]
P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
P2 = K @ np.hstack([R_est, t_est])
in1, in2 = pts1[inliers], pts2[inliers]
X = triangulate(P1, P2, in1, in2)

## Check 1: matches OpenCV (note the transposes and the homogeneous (4, N) output)
X_cv_h = cv.triangulatePoints(P1, P2, in1.T, in2.T)
X_cv = (X_cv_h[:3] / X_cv_h[3]).T
print(f"\ntriangulated {len(X)} inliers")
print(f"max |X - X_opencv|: {np.abs(X - X_cv).max():.2e}")

# ## Check 2: in front of both cameras
z1 = X[:, 2]
z2 = (X @ R_est.T + t_est.ravel())[:, 2]
in_front = (z1 > 0) & (z2 > 0)
print(f"in front of both cameras: {in_front.mean():.1%}")

## Check 3: reprojection error
err1 = np.linalg.norm(project(P1, X) - in1, axis=1)
err2 = np.linalg.norm(project(P2, X) - in2, axis=1)
print(f"reprojection error, median: cam1 {np.median(err1):.3f} px, cam2 {np.median(err2):.3f} px")

## Check 4: plausibility. Units are |t| = 1, so scale by the true step length.
step = np.linalg.norm(t_gt)
X_m = X * step
## Parked car on the right of frame 0 (pixel box picked from the match image)
car = (in1[:, 0] > 760) & (in1[:, 0] < 1020) & (in1[:, 1] > 190) & (in1[:, 1] < 330) & in_front
print(f"right parked car: {car.sum()} points, median depth {np.median(X_m[car, 2]):.1f} m")

## Top-down view: x (right) vs z (forward), only points in front, clipped to a readable range
keep = in_front & (X_m[:, 2] < 80)
fig, ax = plt.subplots(figsize=(7, 8))
ax.scatter(X_m[keep, 0], X_m[keep, 2], s=4, color="tab:blue", alpha=0.6, linewidths=0)
ax.scatter(X_m[car, 0], X_m[car, 2], s=10, color="tab:orange", linewidths=0)
ax.annotate("right parked car", (np.median(X_m[car, 0]), np.median(X_m[car, 2])),
            xytext=(8, 0), textcoords="offset points", fontsize=9)
## Camera centres: C1 = 0, C2 = -R^T t (scaled to metres)
C2 = -R_est.T @ t_est.ravel() * step
ax.scatter([0, C2[0]], [0, C2[2]], marker="^", s=60, color="black", zorder=3)
ax.annotate("cameras", (0, 0), xytext=(8, -4), textcoords="offset points", fontsize=9)
ax.set_aspect("equal")
ax.set_xlabel("x [m] (right)")
ax.set_ylabel("z [m] (forward)")
ax.set_title(f"Triangulated inliers, frame 0 -> 1 (top-down, z < 80 m, n = {keep.sum()})")
ax.grid(True, alpha=0.3)
fig.savefig(RESULTS_DIR / "p1_triangulated_topdown_00_0-1.png", dpi=150, bbox_inches="tight")

# plt.figure(figsize=(14, 5))
# plt.imshow(cv.cvtColor(vis, cv.COLOR_BGR2RGB))
# plt.title(f"ORB matches, frame 0 -> 1: {inliers.sum()} inliers (green), "
#           f"{(~inliers).sum()} outliers (red)")
# plt.axis("off")
# plt.show()
