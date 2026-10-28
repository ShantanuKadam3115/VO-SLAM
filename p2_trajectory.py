import time

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

from vslam.config import RESULTS_DIR, data_root_path
from vslam.features import match_features
from vslam.geometry import estimate_relative_pose
from vslam.kitti_io import load_calib, load_image, load_poses
from vslam.metrics import ate, rotation_angle_deg, rpe
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

MIN_FLOW_PX = 1.5     # median match motion below this = not enough parallax for E
MAX_ROT_DEG = 10.0    # largest real turn is ~4.8 deg/frame; more than this is a bad E

## Fallback motion for frames where pose estimation fails: reuse the last good one
R_est, t_est = np.eye(3), np.zeros((3, 1))
n_failed = n_low_motion = n_implausible = 0
start = time.perf_counter()

## Reference frame: only advances once enough motion has built up since it
ref = 0
ref_img = load_image(seq_dir, ref)

for j in range(1, N):

    img = load_image(seq_dir, j)
    pts1, pts2 = match_features(ref_img, img, orb, bf)

    ## Low-motion check: too little parallax -> keep the reference pose (R = I, t = 0)
    ## and match the next frame against the same reference
    flow = np.median(np.linalg.norm(pts2 - pts1, axis=1)) if len(pts1) else 0.0
    if flow < MIN_FLOW_PX:
        n_low_motion += 1
        estimated_all_poses[j] = estimated_all_poses[ref]
        continue

    ## True step length from the reference to frame j (may span several frames)
    T_rel = np.linalg.inv(gt[ref]) @ gt[j]
    step = np.linalg.norm(T_rel[:3, 3])

    try:
        R_new, t_new, _ = estimate_relative_pose(pts1, pts2, K, threshold=1.0)
        ## Plausibility check: a car can't turn more than a few degrees in 0.1 s
        if rotation_angle_deg(R_new) > MAX_ROT_DEG:
            n_implausible += 1
            print(f"frame {ref}->{j}: rotation {rotation_angle_deg(R_new):.1f} deg; "
                  f"reusing previous motion", flush=True)
        else:
            R_est, t_est = R_new, t_new
    except ValueError as e:
        n_failed += 1
        print(f"frame {ref}->{j}: {e}; reusing previous motion", flush=True)

    t_scaled = step * t_est

    ## X_j = R X_ref + t  =>  T_j = T_ref [R^T | -R^T t]
    estimated_all_poses[j, :3, :3] = estimated_all_poses[ref, :3, :3] @ R_est.T
    estimated_all_poses[j, :3, 3] = estimated_all_poses[ref, :3, 3] - estimated_all_poses[j, :3, :3] @ t_scaled.ravel()

    ref, ref_img = j, img

    if j % 100 == 0:
        elapsed = time.perf_counter() - start
        print(f"{j}/{N - 1}  ({elapsed / 60:.1f} min)", flush=True)

print(f"done in {(time.perf_counter() - start) / 60:.1f} min: {n_low_motion} low-motion frames, "
      f"{n_implausible} implausible rotations, {n_failed} failed frames")

## Save first, so plots/metrics can be redone without rerunning the loop
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
np.save(RESULTS_DIR / f"p2_poses_{seq}.npy", estimated_all_poses)

## Scale comes from GT, so don't use sim3: it would rescale and hide scale drift
rmse_none, _ = ate(estimated_all_poses, gt, align="none")
rmse_se3, aligned_se3 = ate(estimated_all_poses, gt, align="se3")
rpe_t, rpe_r = rpe(estimated_all_poses, gt, delta=1)

## Median per-frame rotation error: the RMSE is dominated by a few bad frames
gt_rel = np.linalg.inv(gt[:-1]) @ gt[1:]
est_rel = np.linalg.inv(estimated_all_poses[:-1]) @ estimated_all_poses[1:]
rot_err = rotation_angle_deg(gt_rel[:, :3, :3].transpose(0, 2, 1) @ est_rel[:, :3, :3])

print(f"Monocular VO with GT scale on KITTI {seq} ({N} frames)")
print(f"  ATE (none): {rmse_none:.2f} m")
print(f"  ATE (se3):  {rmse_se3:.2f} m")
print(f"  RPE (d=1):  {rpe_t:.3f} m, {rpe_r:.3f} deg (median {np.median(rot_err):.3f} deg)")

plot_trajectories([gt, estimated_all_poses], ["ground truth", "VO (GT scale)"],
                  f"KITTI {seq} - monocular VO, GT step length",
                  save_path=RESULTS_DIR / f"p2_trajectory_{seq}.png")
plt.show()
