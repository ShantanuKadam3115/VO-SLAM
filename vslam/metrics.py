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


def rotation_angle_deg(R):
    """Rotation angle in degrees of a (3, 3) or stacked (..., 3, 3) rotation."""
    R = np.asarray(R, dtype=float)
    # trace(R) = 1 + 2 cos(theta); clip guards rounding just outside [-1, 1]
    cos_theta = (np.trace(R, axis1=-2, axis2=-1) - 1) / 2
    return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))


def rpe(est_pose, gt_pose, delta=1):
    """Relative Pose Error over a fixed frame gap (drift per `delta` frames).

    est_pose, gt_pose: (N, 4, 4) camera-to-world poses.
    delta: frame gap between the pose pairs being compared.
    Returns (trans_rmse [m], rot_rmse [deg]).

    RPE ignores the global frame, but not scale: for monocular VO, pass the
    Sim(3)-aligned poses from ate() so the translation error is in metres.
    """
    est_pose = np.asarray(est_pose, dtype=float)
    gt_pose = np.asarray(gt_pose, dtype=float)
    assert est_pose.shape == gt_pose.shape
    assert 0 < delta < est_pose.shape[0]

    # print(est_pose.shape , gt_pose.shape)

    # Relative motion from frame i to i+delta: T_i^-1 T_{i+delta}
    est_rel = np.linalg.inv(est_pose[:-delta]) @ est_pose[delta:]
    gt_rel = np.linalg.inv(gt_pose[:-delta]) @ gt_pose[delta:]

    # Error transform: identity if est moved exactly like gt
    E = np.linalg.inv(gt_rel) @ est_rel


    trans_err = np.linalg.norm(E[:, :3, 3], axis=1)

    rot_err = rotation_angle_deg(E[:, :3, :3])

    trans_rmse = np.sqrt(np.mean(trans_err**2))
    rot_rmse = np.sqrt(np.mean(rot_err**2))
    return trans_rmse, rot_rmse