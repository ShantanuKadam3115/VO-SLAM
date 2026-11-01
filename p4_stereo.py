"""Phase 4: stereo VO. Metric scale from the stereo baseline, no ground truth used.

Each frame: disparity from the left/right pair gives metric 3D points for the
left-image ORB features. The next frame's pose comes from PnP against them.
"""
import time

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from vslam.config import RESULTS_DIR, data_root_path
from vslam.features import detect, match_descriptors
from vslam.kitti_io import load_calib, load_image, load_poses
from vslam.metrics import ate, rotation_angle_deg, rpe
from vslam.plotting import plot_trajectories
from vslam.stereo import backproject, compute_disparity, make_stereo_matcher

## ---- Settings ----
seq = "00"
STEREO_METHOD = "sgbm"   # "bm" for plain block matching
NUM_DISPARITIES = 128    # search range in px (multiple of 16)
MIN_DISP_PX = 1.0        # disparities below this are unreliable
MAX_DEPTH_M = 50.0       # far points constrain translation poorly; depth error grows ~ z^2
MIN_PNP_INLIERS = 30     # fewer than this = tracking failed for this frame
PNP_REPROJ_PX = 2.0      # RANSAC inlier threshold for PnP
MAX_ROT_DEG = 10.0       # reject frame-to-frame rotations a car can't do
MAX_FRAMES = None        # e.g. 300 for a quick test run; None = whole sequence

seq_dir = data_root_path / "sequences" / seq
gt = load_poses(data_root_path / f"poses/{seq}.txt")[:MAX_FRAMES]
K, B = load_calib(seq_dir / "calib.txt")
N = len(gt)
print(f"stereo baseline B = {B:.3f} m, f = {K[0, 0]:.1f} px")

orb = cv.ORB_create(nfeatures=3000)
bf = cv.BFMatcher(cv.NORM_HAMMING)
stereo = make_stereo_matcher(STEREO_METHOD, NUM_DISPARITIES)


def stereo_frame(j):
    """Left-image features of frame j and their metric 3D points (camera frame)."""
    left = load_image(seq_dir, j, cam=0)
    right = load_image(seq_dir, j, cam=1)
    pts, des = detect(left, orb)
    disp = compute_disparity(left, right, stereo)
    X_cam, valid = backproject(pts, disp, K, B, MIN_DISP_PX, MAX_DEPTH_M)
    return pts, des, X_cam, valid


def to_world(X_cam, T):
    """Camera-frame points -> world, with T the camera-to-world pose."""
    return X_cam @ T[:3, :3].T + T[:3, 3]


est = np.tile(np.eye(4), (N, 1, 1))
est[0] = gt[0]                     # only to put the trajectory in GT's frame; no scale info

pts, des, X_cam, valid = stereo_frame(0)
prev_des, prev_X = des[valid], to_world(X_cam[valid], est[0])

vel = np.eye(4)                    # last tracked frame-to-frame motion
stats = {k: np.zeros(N, int) for k in ["landmarks", "matches", "inliers"]}
lost = np.zeros(N, bool)
n_implausible = 0
start = time.perf_counter()

for j in range(1, N):
    pts, des, X_cam, valid = stereo_frame(j)

    ## 2D-3D matches: previous frame's stereo points vs this frame's features
    i_prev, i_cur = match_descriptors(prev_des, des, bf)
    stats["landmarks"][j] = len(prev_X)
    stats["matches"][j] = len(i_prev)

    T = None
    if len(i_prev) >= MIN_PNP_INLIERS:
        obj = prev_X[i_prev].astype(np.float64)
        img = pts[i_cur].astype(np.float64)
        ok, rvec, tvec, inl = cv.solvePnPRansac(obj, img, K, None, iterationsCount=200,
                                               reprojectionError=PNP_REPROJ_PX, confidence=0.999)
        if ok and inl is not None and len(inl) >= MIN_PNP_INLIERS:
            inl = inl.ravel()
            ## Polish on the inliers; rvec, tvec map world -> camera
            rvec, tvec = cv.solvePnPRefineLM(obj[inl], img[inl], K, None, rvec, tvec)
            T_wc = np.eye(4)
            T_wc[:3, :3] = cv.Rodrigues(rvec)[0]
            T_wc[:3, 3] = tvec.ravel()
            T = np.linalg.inv(T_wc)
            stats["inliers"][j] = len(inl)

    ## Plausibility check, as in Phase 2: a car can't turn > MAX_ROT_DEG in 0.1 s
    if T is not None and rotation_angle_deg(est[j - 1][:3, :3].T @ T[:3, :3]) > MAX_ROT_DEG:
        n_implausible += 1
        T = None

    if T is None:
        ## Constant-velocity fallback
        lost[j] = True
        est[j] = est[j - 1] @ vel
    else:
        est[j] = T
        vel = np.linalg.inv(est[j - 1]) @ est[j]

    ## This frame's stereo points become the landmarks for the next frame
    prev_des, prev_X = des[valid], to_world(X_cam[valid], est[j])

    if j % 100 == 0:
        print(f"{j}/{N - 1}  ({(time.perf_counter() - start) / 60:.1f} min)  "
              f"inliers {stats['inliers'][j]}", flush=True)

print(f"done in {(time.perf_counter() - start) / 60:.1f} min: {lost.sum()} lost frames "
      f"({n_implausible} implausible rotations)")

## Save first, so investigation doesn't need a rerun
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
np.save(RESULTS_DIR / f"p4_poses_{seq}.npy", est)
np.savez(RESULTS_DIR / f"p4_stats_{seq}.npz", lost=lost, **stats)

## ---- Evaluation ----
## Stereo is metric, so SE(3) is the honest alignment; Sim(3) shows how much
## a global scale correction would still help (it shouldn't, much).
rmse_se3, _ = ate(est, gt, align="se3")
rmse_sim3, _ = ate(est, gt, align="sim3")
rmse_none, _ = ate(est, gt, align="none")
rpe_t, rpe_r = rpe(est, gt, delta=1)

gt_rel = np.linalg.inv(gt[:-1]) @ gt[1:]
est_rel = np.linalg.inv(est[:-1]) @ est[1:]
rot_err = rotation_angle_deg(gt_rel[:, :3, :3].transpose(0, 2, 1) @ est_rel[:, :3, :3])

gt_step = np.linalg.norm(gt_rel[:, :3, 3], axis=1)
est_step = np.linalg.norm(est_rel[:, :3, 3], axis=1)
moving = gt_step > 0.05
scale_ratio = np.full(N - 1, np.nan)
scale_ratio[moving] = est_step[moving] / gt_step[moving]

print(f"Stereo VO ({STEREO_METHOD}) on KITTI {seq} ({N} frames), no GT scale")
print(f"  ATE (se3):  {rmse_se3:.2f} m   ATE (sim3): {rmse_sim3:.2f} m   ATE (none): {rmse_none:.2f} m")
print(f"  RPE (d=1):  {rpe_t:.3f} m, {rpe_r:.3f} deg (median {np.median(rot_err):.3f} deg)")
print(f"  scale ratio (est/GT step), median over quarters of the run: "
      f"{[round(float(np.nanmedian(q)), 3) for q in np.array_split(scale_ratio, 4)]}")

plot_trajectories([gt, est], ["ground truth", f"stereo VO ({STEREO_METHOD})"],
                  f"KITTI {seq} - stereo VO, no ground truth",
                  save_path=RESULTS_DIR / f"p4_trajectory_{seq}.png")

## ---- Checkpoint: mono vs stereo vs GT, side by side ----
## Mono runs are shown as saved (P2 borrows GT scale every frame) and Sim(3)-aligned
## (P3 has no metric scale, so raw it would be a dot). Stereo is shown raw.
panels = [("P2 mono, GT step length", "p2", None), ("P3 mono, PnP (Sim(3)-aligned)", "p3", "sim3"),
          (f"P4 stereo ({STEREO_METHOD}), raw", None, None)]
fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
for ax, (label, name, align) in zip(axes, panels):
    if name is None:
        traj = est
    else:
        path = RESULTS_DIR / f"{name}_poses_{seq}.npy"
        if not path.exists():
            ax.set_title(f"{label}: no saved poses")
            ax.axis("off")
            continue
        traj = np.load(path)[:N]
        if align:
            _, traj = ate(traj, gt, align=align)
    err, _ = ate(traj, gt, align="none")
    plot_trajectories([gt, traj], ["ground truth", label], f"{label}\nATE as plotted: {err:.1f} m", ax=ax)
fig.tight_layout()
fig.savefig(RESULTS_DIR / f"p4_mono_vs_stereo_{seq}.png", dpi=150, bbox_inches="tight")
plt.show()
