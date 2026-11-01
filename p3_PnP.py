"""Phase 3: monocular VO with PnP against a local 3D map.

Ground truth is used ONCE, for the scale of the initial two-view map.
After that, every pose comes from PnP against the map, and new landmarks are
triangulated from those poses, so scale propagates through the map.
"""
import time

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from vslam.config import RESULTS_DIR, data_root_path
from vslam.features import detect, match_by_projection, match_descriptors
from vslam.geometry import estimate_relative_pose, parallax_deg, project, triangulate
from vslam.kitti_io import load_calib, load_image, load_poses
from vslam.metrics import ate, rotation_angle_deg, rpe
from vslam.plotting import plot_trajectories

## ---- Settings ----
seq = "00"
INIT_GAP = 5             # initial map from frames 0 and INIT_GAP (~4 m baseline)
KF_MIN_GAP = 3           # never triangulate over fewer frames than this (short baseline = bad depth)
KF_MAX_GAP = 5           # make a keyframe at least every this many frames...
KF_MIN_INLIERS = 100     # ...or sooner (but after KF_MIN_GAP), when PnP inliers drop below this
MIN_PNP_INLIERS = 30     # fewer than this = tracking failed for this frame
PNP_REPROJ_PX = 3.0      # RANSAC inlier threshold for PnP (landmark depths are noisy without BA)
SEARCH_RADIUS_PX = 15.0  # projection-guided matching window around the predicted position
TRI_REPROJ_PX = 2.0      # max reprojection error for a new landmark, in both views
MIN_PARALLAX_DEG = 0.5   # min ray angle for a new landmark; 1 deg left too few points to track
CULL_AFTER = 20          # drop landmarks not seen as PnP inliers for this many frames
MAX_OUTLIER_COUNT = 3    # drop landmarks that were PnP outliers this often (and more than inliers)
MAX_ROT_DEG = 10.0       # reject frame-to-frame rotations a car can't do
REINIT_AFTER_LOST = 2    # consecutive lost frames before re-initializing from two views
MAX_FRAMES = None        # e.g. 300 for a quick test run; None = whole sequence

seq_dir = data_root_path / "sequences" / seq
gt = load_poses(data_root_path / f"poses/{seq}.txt")[:MAX_FRAMES]
K, _ = load_calib(seq_dir / "calib.txt")
N = len(gt)
H, W = load_image(seq_dir, 0).shape

orb = cv.ORB_create(nfeatures=3000)
bf = cv.BFMatcher(cv.NORM_HAMMING)


def world_to_cam(T):
    """(4, 4) camera-to-world pose -> (3, 4) [R | t] with X_cam = R X_world + t."""
    return np.linalg.inv(T)[:3, :]


def triangulate_new(T1, T2, pts1, pts2):
    """Triangulate matches between two posed frames; return (X (N, 3), good mask)."""
    Rt1, Rt2 = world_to_cam(T1), world_to_cam(T2)
    P1, P2 = K @ Rt1, K @ Rt2
    X = triangulate(P1, P2, pts1, pts2)

    z1 = (X @ Rt1[:, :3].T + Rt1[:, 3])[:, 2]
    z2 = (X @ Rt2[:, :3].T + Rt2[:, 3])[:, 2]
    err1 = np.linalg.norm(project(P1, X) - pts1, axis=1)
    err2 = np.linalg.norm(project(P2, X) - pts2, axis=1)
    par = parallax_deg(X, T1[:3, 3], T2[:3, 3])

    good = (z1 > 0) & (z2 > 0) & (err1 < TRI_REPROJ_PX) & (err2 < TRI_REPROJ_PX) & (par > MIN_PARALLAX_DEG)
    return X, good


def two_view_map(T1, pts1, des1, pts2, des2, step):
    """Pose of view 2 from E (with |t| = step) and a fresh map from the two views.

    Returns (T2, X, des of landmarks, feature idx used in view 1, in view 2).
    Raises ValueError if E can't be estimated.
    """
    i1, i2 = match_descriptors(des1, des2, bf)
    R, t, inl = estimate_relative_pose(pts1[i1], pts2[i2], K)

    T_rel = np.eye(4)
    T_rel[:3, :3] = R.T                      # X_2 = R X_1 + t  =>  T_2 = T_1 [R^T | -R^T t]
    T_rel[:3, 3] = -R.T @ (step * t.ravel())
    T2 = T1 @ T_rel

    i1, i2 = i1[inl], i2[inl]
    X, good = triangulate_new(T1, T2, pts1[i1], pts2[i2])
    return T2, X[good], des2[i2[good]], i1[good], i2[good]


def track(pts, des, T_pred):
    """PnP + RANSAC of one frame against the map, matching around predicted projections.

    Returns (T (4, 4) camera-to-world, matched map idx, matched feature idx,
    inlier mask over those matches), or None if tracking failed.
    """
    ## Predict where each map point should appear; only search near there
    Rt = world_to_cam(T_pred)
    z = map_X @ Rt[:, :3].T[:, 2] + Rt[2, 3]
    proj = project(K @ Rt, map_X)
    visible = (z > 0) & (proj[:, 0] >= 0) & (proj[:, 0] < W) & (proj[:, 1] >= 0) & (proj[:, 1] < H)
    vis_idx = np.where(visible)[0]

    m, feat_idx = match_by_projection(proj[vis_idx], map_des[vis_idx], pts, des, bf,
                                      radius=SEARCH_RADIUS_PX)
    map_idx = vis_idx[m]
    ## Bad prediction (e.g. after a re-init): fall back to global descriptor matching
    if len(map_idx) < MIN_PNP_INLIERS:
        map_idx, feat_idx = match_descriptors(map_des, des, bf)
    if len(map_idx) < MIN_PNP_INLIERS:
        return None

    obj = map_X[map_idx].astype(np.float64)
    img = pts[feat_idx].astype(np.float64)
    ok, rvec, tvec, inl = cv.solvePnPRansac(obj, img, K, None, iterationsCount=200,
                                           reprojectionError=PNP_REPROJ_PX, confidence=0.999)
    if not ok or inl is None or len(inl) < MIN_PNP_INLIERS:
        return None
    inl = inl.ravel()

    ## Polish the RANSAC pose on all inliers (Levenberg-Marquardt on reprojection error)
    rvec, tvec = cv.solvePnPRefineLM(obj[inl], img[inl], K, None, rvec, tvec)

    ## rvec, tvec map world -> camera; invert to get the camera-to-world pose
    T_wc = np.eye(4)
    T_wc[:3, :3] = cv.Rodrigues(rvec)[0]
    T_wc[:3, 3] = tvec.ravel()

    inlier_mask = np.zeros(len(map_idx), bool)
    inlier_mask[inl] = True
    return np.linalg.inv(T_wc), map_idx, feat_idx, inlier_mask


def new_map(X, des, j):
    """Fresh map arrays: (X, des, last seen, n inlier, n outlier)."""
    n = len(X)
    return X, des, np.full(n, j), np.ones(n, int), np.zeros(n, int)


## ---- Initialization: two-view map from frames 0 and INIT_GAP ----
est = np.tile(np.eye(4), (N, 1, 1))
est[0] = gt[0]

pts0, des0 = detect(load_image(seq_dir, 0), orb)
ptsk, desk = detect(load_image(seq_dir, INIT_GAP), orb)

## The ONLY use of ground truth: the length of the initial baseline
step = np.linalg.norm((np.linalg.inv(gt[0]) @ gt[INIT_GAP])[:3, 3])
est[INIT_GAP], X, X_des, _, used_k = two_view_map(est[0], pts0, des0, ptsk, desk, step)

## The map: one row per landmark
map_X, map_des, map_seen, map_n_in, map_n_out = new_map(X, X_des, INIT_GAP)
print(f"init: {len(map_X)} landmarks from frames 0 and {INIT_GAP}")

## Frames between the two init views: localize only (pose predicted by interpolation)
for j in range(1, INIT_GAP):
    pts, des = detect(load_image(seq_dir, j), orb)
    T_pred = est[0].copy()
    T_pred[:3, 3] = est[0][:3, 3] + (est[INIT_GAP][:3, 3] - est[0][:3, 3]) * j / INIT_GAP
    res = track(pts, des, T_pred)
    est[j] = res[0] if res is not None else T_pred

## Last keyframe: its features, and which of them are already landmarks
kf_idx, kf_pts, kf_des = INIT_GAP, ptsk, desk
kf_used = np.zeros(len(ptsk), bool)
kf_used[used_k] = True

## Per-frame stats for investigation
stats = {k: np.zeros(N, int) for k in ["matches", "inliers", "map_size", "new_landmarks"]}
lost = np.zeros(N, bool)
reinit = np.zeros(N, bool)
n_implausible = 0
n_lost_in_a_row = 0
## Per-frame motion (camera-to-camera), used to predict the next pose. Only updated
## from tracked frames or a re-init, so lost frames can't compound their own guesses.
vel = np.linalg.inv(est[INIT_GAP - 1]) @ est[INIT_GAP]
start = time.perf_counter()

## ---- Main loop ----
for j in range(INIT_GAP + 1, N):
    pts, des = detect(load_image(seq_dir, j), orb)

    ## Constant-velocity prediction: used for projection matching and as the fallback pose
    T_pred = est[j - 1] @ vel
    res = track(pts, des, T_pred)

    ## Plausibility check, as in Phase 2: a car can't turn > MAX_ROT_DEG in 0.1 s
    if res is not None:
        turn = rotation_angle_deg(est[j - 1][:3, :3].T @ res[0][:3, :3])
        if turn > MAX_ROT_DEG:
            n_implausible += 1
            print(f"frame {j}: PnP rotation {turn:.1f} deg; rejected", flush=True)
            res = None

    feat_used = np.zeros(len(pts), bool)
    n_inliers = 0

    if res is None:
        lost[j] = True
        n_lost_in_a_row += 1
        est[j] = T_pred

        ## Re-initialize from two views (last keyframe and this frame). The scale comes
        ## from the current velocity (last tracked motion), not from ground truth.
        if n_lost_in_a_row >= REINIT_AFTER_LOST:
            gap = j - kf_idx
            step = np.linalg.norm(vel[:3, 3]) * gap
            try:
                T_j, X, X_des, _, used = two_view_map(est[kf_idx], kf_pts, kf_des, pts, des, step)
            except ValueError:
                T_j, X = None, []
            if len(X) >= MIN_PNP_INLIERS:
                ## New velocity: the re-init motion spread evenly over the gap (rotation ignored)
                vel = np.eye(4)
                vel[:3, 3] = (np.linalg.inv(est[kf_idx]) @ T_j)[:3, 3] / gap
                est[j] = T_j
                map_X, map_des, map_seen, map_n_in, map_n_out = new_map(X, X_des, j)
                feat_used[used] = True
                kf_idx, kf_pts, kf_des, kf_used = j, pts, des, feat_used
                reinit[j] = True
                n_lost_in_a_row = 0
                print(f"frame {j}: re-initialized, {len(X)} landmarks", flush=True)
            else:
                ## Failed: make this frame the keyframe so the next attempt has a short
                ## gap, instead of matching against an ever older frame
                kf_idx, kf_pts, kf_des, kf_used = j, pts, des, np.zeros(len(pts), bool)
    else:
        n_lost_in_a_row = 0
        est[j], map_idx, feat_idx, inlier_mask = res
        if not lost[j - 1]:
            vel = np.linalg.inv(est[j - 1]) @ est[j]
        n_inliers = inlier_mask.sum()
        stats["matches"][j] = len(map_idx)

        ## Book-keeping: inliers refresh their descriptor (appearance changes as we drive)
        in_m, out_m = map_idx[inlier_mask], map_idx[~inlier_mask]
        map_seen[in_m] = j
        map_n_in[in_m] += 1
        map_n_out[out_m] += 1
        map_des[in_m] = des[feat_idx[inlier_mask]]
        feat_used[feat_idx[inlier_mask]] = True

    stats["inliers"][j] = n_inliers

    ## ---- Keyframe: triangulate new landmarks against the previous keyframe ----
    gap = j - kf_idx
    if n_inliers > 0 and gap >= KF_MIN_GAP and (n_inliers < KF_MIN_INLIERS or gap >= KF_MAX_GAP):
        ## Only features that aren't landmarks yet, in both frames
        kf_free = np.where(~kf_used)[0]
        cur_free = np.where(~feat_used)[0]
        a, b = match_descriptors(kf_des[kf_free], des[cur_free], bf)
        a, b = kf_free[a], cur_free[b]

        n_new = 0
        if len(a):
            X, good = triangulate_new(est[kf_idx], est[j], kf_pts[a], pts[b])
            n_new = good.sum()
            map_X = np.vstack([map_X, X[good]])
            map_des = np.vstack([map_des, des[b[good]]])
            map_seen = np.concatenate([map_seen, np.full(n_new, j)])
            map_n_in = np.concatenate([map_n_in, np.ones(n_new, int)])
            map_n_out = np.concatenate([map_n_out, np.zeros(n_new, int)])
            feat_used[b[good]] = True
        stats["new_landmarks"][j] = n_new

        kf_idx, kf_pts, kf_des, kf_used = j, pts, des, feat_used

    ## ---- Cull: keep the map local and drop landmarks that keep failing PnP ----
    keep = (j - map_seen <= CULL_AFTER) & ~((map_n_out >= MAX_OUTLIER_COUNT) & (map_n_out > map_n_in))
    map_X, map_des = map_X[keep], map_des[keep]
    map_seen, map_n_in, map_n_out = map_seen[keep], map_n_in[keep], map_n_out[keep]
    stats["map_size"][j] = len(map_X)

    if j % 100 == 0:
        print(f"{j}/{N - 1}  ({(time.perf_counter() - start) / 60:.1f} min)  "
              f"inliers {n_inliers}, map {len(map_X)}", flush=True)

print(f"done in {(time.perf_counter() - start) / 60:.1f} min: {lost.sum()} lost frames "
      f"({n_implausible} implausible rotations), {reinit.sum()} re-initializations")

## Save first, so investigation doesn't need a rerun
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
np.save(RESULTS_DIR / f"p3_poses_{seq}.npy", est)
np.savez(RESULTS_DIR / f"p3_stats_{seq}.npz", lost=lost, reinit=reinit, **stats)

## ---- Evaluation ----
## Scale is no longer borrowed per frame, so Sim(3) is now a fair alignment;
## SE(3) shows how much scale drift costs.
rmse_sim3, _ = ate(est, gt, align="sim3")
rmse_se3, _ = ate(est, gt, align="se3")
rpe_t, rpe_r = rpe(est, gt, delta=1)

gt_rel = np.linalg.inv(gt[:-1]) @ gt[1:]
est_rel = np.linalg.inv(est[:-1]) @ est[1:]
rot_err = rotation_angle_deg(gt_rel[:, :3, :3].transpose(0, 2, 1) @ est_rel[:, :3, :3])

## Scale drift: estimated / true step length per frame (1.0 = no drift)
gt_step = np.linalg.norm(gt_rel[:, :3, 3], axis=1)
est_step = np.linalg.norm(est_rel[:, :3, 3], axis=1)
moving = gt_step > 0.05
scale_ratio = np.full(N - 1, np.nan)
scale_ratio[moving] = est_step[moving] / gt_step[moving]

print(f"Monocular VO + PnP local map on KITTI {seq} ({N} frames), GT scale at init only")
print(f"  ATE (sim3): {rmse_sim3:.2f} m")
print(f"  ATE (se3):  {rmse_se3:.2f} m")
print(f"  RPE (d=1):  {rpe_t:.3f} m, {rpe_r:.3f} deg (median {np.median(rot_err):.3f} deg)")
print(f"  scale ratio (est/GT step), median over quarters of the run: "
      f"{[round(float(np.nanmedian(q)), 2) for q in np.array_split(scale_ratio, 4)]}")

plot_trajectories([gt, est], ["ground truth", "VO + PnP (GT scale at init)"],
                  f"KITTI {seq} - monocular VO with PnP and a local map",
                  save_path=RESULTS_DIR / f"p3_trajectory_{seq}.png")

## Scale drift and tracking health over the sequence, one measure per panel
frames = np.arange(N)
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
axes[0].plot(frames[:-1], scale_ratio, linewidth=0.6, color="tab:blue")
axes[0].axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
axes[0].set_ylabel("est / GT step")
axes[0].set_yscale("log")
axes[1].plot(frames, stats["inliers"], linewidth=0.6, color="tab:blue")
axes[1].scatter(frames[lost], np.zeros(lost.sum()), s=10, color="tab:red", zorder=3,
                label=f"lost ({lost.sum()} frames)")
axes[1].scatter(frames[reinit], np.zeros(reinit.sum()), s=30, marker="^", color="black", zorder=4,
                label=f"re-initialized ({reinit.sum()})")
axes[1].set_ylabel("PnP inliers")
axes[1].legend(loc="upper right")
axes[2].plot(frames, stats["map_size"], linewidth=0.6, color="tab:blue")
axes[2].set_ylabel("map size")
axes[2].set_xlabel("frame")
for ax in axes:
    ax.grid(True, alpha=0.3)
fig.suptitle(f"KITTI {seq} - Phase 3 scale drift and tracking")
fig.tight_layout()
fig.savefig(RESULTS_DIR / f"p3_health_{seq}.png", dpi=150, bbox_inches="tight")
plt.show()
