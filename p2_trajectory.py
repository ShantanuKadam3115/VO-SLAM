import time

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

from vslam.config import RESULTS_DIR, data_root_path
from vslam.features import match_features
from vslam.geometry import estimate_relative_pose
from vslam.kitti_io import load_calib, load_image, load_poses
from vslam.metrics import ate, rpe
from vslam.plotting import plot_trajectories

seq = "00"
seq_dir = data_root_path / "sequences" / seq
calib_dir = seq_dir / "calib.txt"
gt = load_poses(data_root_path / f"poses/{seq}.txt")
K, _ = load_calib(calib_dir)
N = len(gt)

## Identity stack so every pose has a valid [0, 0, 0, 1] bottom row
estimated_all_poses = np.tile(np.eye(4), (N, 1, 1))
estimated_all_poses[0] = gt[0]

orb = cv.ORB_create(nfeatures=3000)
bf = cv.BFMatcher(cv.NORM_HAMMING)

R_est, t_est = np.eye(3), np.zeros((3, 1))
n_failed = 0
start = time.perf_counter()

for i in range(N - 1):

    T_rel = np.linalg.inv(gt[i]) @ gt[i+1]
    step = np.linalg.norm(T_rel[:3, 3])

    img1 = load_image(seq_dir, i)
    img2 = load_image(seq_dir, i+1)

    pts1, pts2 = match_features(img1, img2, orb, bf)

    try:
        R_est, t_est, _ = estimate_relative_pose(pts1, pts2, K, threshold=1.0)
    except ValueError as e:
        n_failed += 1
        print(f"frame {i}: {e}; reusing previous motion", flush=True)

    t_scaled = step * t_est

    estimated_all_poses[i+1, :3, :3] = estimated_all_poses[i, :3, :3] @ R_est.T
    estimated_all_poses[i+1, :3, 3] = estimated_all_poses[i, :3, 3] - estimated_all_poses[i+1, :3, :3] @ t_scaled.ravel()

    if i % 100 == 0:
        elapsed = time.perf_counter() - start
        print(f"{i}/{N - 1}  ({elapsed / 60:.1f} min)", flush=True)

print(f"done in {(time.perf_counter() - start) / 60:.1f} min, {n_failed} failed frames")

## Save first, so plots/metrics can be redone without rerunning the loop
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
np.save(RESULTS_DIR / f"p2_poses_{seq}.npy", estimated_all_poses)

## Scale comes from GT, so don't use sim3: it would rescale and hide scale drift
rmse_none, _ = ate(estimated_all_poses, gt, align="none")
rmse_se3, aligned_se3 = ate(estimated_all_poses, gt, align="se3")
rpe_t, rpe_r = rpe(estimated_all_poses, gt, delta=1)

print(f"Monocular VO with GT scale on KITTI {seq} ({N} frames)")
print(f"  ATE (none): {rmse_none:.2f} m")
print(f"  ATE (se3):  {rmse_se3:.2f} m")
print(f"  RPE (d=1):  {rpe_t:.3f} m, {rpe_r:.3f} deg")

plot_trajectories([gt, estimated_all_poses], ["ground truth", "VO (GT scale)"],
                  f"KITTI {seq} - monocular VO, GT step length",
                  save_path=RESULTS_DIR / f"p2_trajectory_{seq}.png")
plt.show()
