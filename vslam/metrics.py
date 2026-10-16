import numpy as np


def umeyama(est, gt, with_scale=True):
    """Find s, R, t minimising sum ||gt_i - (s R est_i + t)||^2.

    est, gt: (N, 3) corresponding points.
    Umeyama 1991, eqs. 40-42.
    """
    est = np.asarray(est, dtype=float)
    gt = np.asarray(gt, dtype=float)
    assert est.shape == gt.shape and est.shape[1] == 3

    N = est.shape[0]
    mu_est = est.mean(axis=0)
    mu_gt = gt.mean(axis=0)
    est_c = est - mu_est
    gt_c = gt - mu_gt

    # Cross-covariance (3x3): gt on the left because we map est -> gt
    Sigma = gt_c.T @ est_c / N

    U, D, Vt = np.linalg.svd(Sigma)

    # Reflection guard: without it, R could have det = -1 (a mirror)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt

    if with_scale:
        var_est = np.mean(np.sum(est_c**2, axis=1))
        if var_est < 1e-12:
            # All est points are the same (e.g. a zero-motion baseline):
            # there is nothing to scale, so the scale is undefined.
            raise ValueError("est has ~zero spread; Sim(3) scale is undefined")
        s = np.trace(np.diag(D) @ S) / var_est
    else:
        s = 1.0

    t = mu_gt - s * R @ mu_est
    return s, R, t


def ate(est_poses, gt_poses, align="sim3"):
    """Absolute Trajectory Error (translation RMSE) after alignment.

    est_poses, gt_poses: (N, 4, 4) camera-to-world poses.
    align: "sim3" (scale+rot+trans), "se3" (rot+trans), or "none".
    Returns (rmse, aligned_est_poses) where aligned_est_poses is (N, 4, 4).
    """
    est_poses = np.asarray(est_poses, dtype=float)
    gt_poses = np.asarray(gt_poses, dtype=float)
    assert est_poses.shape == gt_poses.shape

    est_xyz = est_poses[:, :3, 3]
    gt_xyz = gt_poses[:, :3, 3]

    
    if align == "sim3":
        s, R, t = umeyama(est_xyz, gt_xyz, with_scale=True)
    elif align == "se3":
        s, R, t = umeyama(est_xyz, gt_xyz, with_scale=False)
    elif align == "none":
        s, R, t = 1.0, np.eye(3), np.zeros(3)
    else:
        raise ValueError(f"unknown align mode: {align!r}")

    # Apply the same transform to every pose: positions get s R p + t,
    # orientations get R R_i (scale doesn't touch rotations).
    aligned = est_poses.copy()
    aligned[:, :3, 3] = s * est_xyz @ R.T + t
    aligned[:, :3, :3] = R @ est_poses[:, :3, :3]

    err = gt_xyz - aligned[:, :3, 3]
    rmse = np.sqrt(np.mean(np.sum(err**2, axis=1)))
    return rmse, aligned
