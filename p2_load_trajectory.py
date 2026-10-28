"""Investigate a saved Phase 2 trajectory without rerunning the VO loop.

Edit the settings below, run, and look at the printed table and the figures.
"""
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from vslam.config import RESULTS_DIR, data_root_path
from vslam.features import match_features
from vslam.geometry import estimate_relative_pose
from vslam.kitti_io import load_calib, load_image, load_poses
from vslam.metrics import ate, rotation_angle_deg, rpe
from vslam.plotting import plot_trajectories

## ---- Settings ----
seq = "00"
BAD_ROT_DEG = 2.0         # a frame counts as "bad" if its rotation error is above this
STOPPED_STEP_M = 0.05     # GT step below this = car (nearly) stopped
LABEL_GAP = 10            # bad frames closer than this many frames share one label in figure 2
LABEL_RADIUS_M = 20.0     # ...and so do label groups closer than this in space
N_WORST = 25             # rows in the worst-frames table
FRAMES_TO_INSPECT = [154, 364, 893]    # e.g. [123, 456]: re-run matching on these pairs and show the arrows

seq_dir = data_root_path / "sequences" / seq
gt = load_poses(data_root_path / f"poses/{seq}.txt")
est = np.load(RESULTS_DIR / f"p2_poses_{seq}.npy")
assert est.shape == gt.shape
N = len(gt)

## ---- Overall metrics (same as p2_trajectory.py) ----
rmse_none, _ = ate(est, gt, align="none")
rmse_se3, _ = ate(est, gt, align="se3")
rpe_t, rpe_r = rpe(est, gt, delta=1)
print(f"KITTI {seq}, {N} frames")
print(f"  ATE (none): {rmse_none:.2f} m   ATE (se3): {rmse_se3:.2f} m")
print(f"  RPE (d=1):  {rpe_t:.3f} m, {rpe_r:.3f} deg")

## ---- Per-frame errors: frame i -> i+1 ----
## Relative motion in camera-i coordinates (pose of camera i+1 seen from camera i)
gt_rel = np.linalg.inv(gt[:-1]) @ gt[1:]
est_rel = np.linalg.inv(est[:-1]) @ est[1:]

rot_err = rotation_angle_deg(gt_rel[:, :3, :3].transpose(0, 2, 1) @ est_rel[:, :3, :3])

gt_turn = rotation_angle_deg(gt_rel[:, :3, :3])      # how much the car actually turned
gt_step = np.linalg.norm(gt_rel[:, :3, 3], axis=1)   # how far the car actually moved

## Translation direction error; undefined when the car isn't moving
t_gt = gt_rel[:, :3, 3]
t_est = est_rel[:, :3, 3]
cos = np.sum(t_gt * t_est, axis=1) / (np.linalg.norm(t_gt, axis=1) * np.linalg.norm(t_est, axis=1) + 1e-12)
dir_err = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
dir_err[gt_step < STOPPED_STEP_M] = np.nan

bad = rot_err > BAD_ROT_DEG
stopped = gt_step < STOPPED_STEP_M
print(f"\nper-frame rotation error: median {np.median(rot_err):.3f} deg, "
      f"90th pct {np.percentile(rot_err, 90):.3f} deg, max {rot_err.max():.1f} deg")
print(f"bad frames (> {BAD_ROT_DEG} deg): {bad.sum()}, of which stopped: {(bad & stopped).sum()}")
print(f"rotation RMS without bad frames: {np.sqrt(np.mean(rot_err[~bad] ** 2)):.3f} deg")

## ---- Worst frames ----
worst = np.argsort(rot_err)[::-1][:N_WORST]
print(f"\nworst {N_WORST} frames (pair i -> i+1):")
print(f"{'frame':>6} {'rot err':>9} {'GT turn':>8} {'GT step':>8} {'t dir err':>9}  note")
for i in worst:
    note = "stopped" if stopped[i] else ("turning" if gt_turn[i] > 1.0 else "")
    print(f"{i:6d} {rot_err[i]:8.2f}° {gt_turn[i]:7.2f}° {gt_step[i]:7.3f}m {dir_err[i]:8.2f}°  {note}")

## ---- Figure 1: per-frame diagnostics, one measure per panel, shared frame axis ----
frames = np.arange(N - 1)
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

axes[0].plot(frames, rot_err, linewidth=0.8, color="tab:blue")
axes[0].scatter(frames[bad], rot_err[bad], s=12, color="tab:red", zorder=3,
                label=f"> {BAD_ROT_DEG}° ({bad.sum()} frames)")
axes[0].set_yscale("log")
axes[0].set_ylabel("rotation error [deg]")
axes[0].legend(loc="upper right")

axes[1].plot(frames, gt_step, linewidth=0.8, color="tab:blue")
axes[1].axhline(STOPPED_STEP_M, color="gray", linestyle="--", linewidth=0.8)
axes[1].set_ylabel("GT step [m]")

axes[2].plot(frames, gt_turn, linewidth=0.8, color="tab:blue")
axes[2].set_ylabel("GT turn [deg]")
axes[2].set_xlabel("frame i (pair i -> i+1)")

for ax in axes:
    ax.grid(True, alpha=0.3)
fig.suptitle(f"KITTI {seq} - per-frame VO errors vs. car motion")
fig.tight_layout()
fig.savefig(RESULTS_DIR / f"p2_frame_errors_{seq}.png", dpi=150, bbox_inches="tight")

## ---- Figure 2: trajectory with the bad frames marked on the GT path ----
ax = plot_trajectories([gt, est], ["ground truth", "VO (GT scale)"],
                       f"KITTI {seq} - bad frames (rotation error > {BAD_ROT_DEG}°)")
ax.scatter(gt[:-1][bad, 0, 3], gt[:-1][bad, 2, 3], s=30, facecolors="none",
           edgecolors="tab:red", linewidths=1.2, zorder=4, label="bad frame")

## Label each group of nearby bad frames once ("547–559"), so a stop doesn't print 13 numbers
groups = []
for i in np.where(bad)[0]:
    if groups and i - groups[-1][-1] <= LABEL_GAP:
        groups[-1].append(i)
    else:
        groups.append([i])
## The route revisits intersections, so also merge groups that are close in space
## into one multi-line label (anchored at the first group's position)
spots = []  # [(x, z), [label texts]]
for g in groups:
    text = str(g[0]) if len(g) == 1 else f"{g[0]}–{g[-1]}"
    xz = gt[g[0], [0, 2], 3]
    for anchor, texts in spots:
        if np.linalg.norm(xz - anchor) < LABEL_RADIUS_M:
            texts.append(text)
            break
    else:
        spots.append((xz, [text]))
for (x, z), texts in spots:
    ax.annotate("\n".join(texts), (x, z), xytext=(5, 4), textcoords="offset points",
                fontsize=7, color="black", va="bottom", zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
ax.legend()
ax.figure.savefig(RESULTS_DIR / f"p2_bad_frames_{seq}.png", dpi=150, bbox_inches="tight")

# ---- Optional: re-run one frame pair and look at its matches ----
if FRAMES_TO_INSPECT:
    K, _ = load_calib(seq_dir / "calib.txt")
    orb = cv.ORB_create(nfeatures=3000)
    bf = cv.BFMatcher(cv.NORM_HAMMING)

for i in FRAMES_TO_INSPECT:
    img1 = load_image(seq_dir, i)
    img2 = load_image(seq_dir, i + 1)
    pts1, pts2 = match_features(img1, img2, orb, bf)
    _, _, inliers = estimate_relative_pose(pts1, pts2, K, threshold=1.0)
    flow = np.median(np.linalg.norm(pts2 - pts1, axis=1))

    vis = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
    for (x0, y0), (x1, y1), ok in zip(pts1, pts2, inliers):
        color = (0, 255, 0) if ok else (0, 0, 255)
        cv.arrowedLine(vis, (round(x0), round(y0)), (round(x1), round(y1)),
                       color, 1, tipLength=0.3)

    plt.figure(figsize=(14, 5))
    plt.imshow(cv.cvtColor(vis, cv.COLOR_BGR2RGB))
    plt.title(f"frame {i} -> {i + 1}: rot err {rot_err[i]:.2f}°, GT step {gt_step[i]:.3f} m, "
              f"{inliers.sum()}/{len(pts1)} inliers, median flow {flow:.1f} px")
    plt.axis("off")

plt.show()
