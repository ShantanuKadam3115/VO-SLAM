import cv2
import numpy as np

from vslam.config import data_root_path
from vslam.kitti_io import load_poses
from vslam.metrics import ate, rpe, umeyama


def make_distorted_est(gt):
    """Apply a known Sim(3) to GT, then return (est, s, R, t) where
    est is what umeyama must map back onto gt."""
    s_true = 2.5
    R_true, _ = cv2.Rodrigues(np.array([0.3, -0.5, 0.8]))
    t_true = np.array([10.0, -4.0, 7.0])

    # gt = s R est + t  =>  est = R^T (gt - t) / s
    est = gt.copy()
    est[:, :3, 3] = (gt[:, :3, 3] - t_true) @ R_true / s_true
    est[:, :3, :3] = R_true.T @ gt[:, :3, :3]
    return est, s_true, R_true, t_true


def test_umeyama_recovers_known_transform():
    gt = load_poses(data_root_path / "poses/00.txt")
    est, s_true, R_true, t_true = make_distorted_est(gt)

    # print(gt[:, :3, 3].shape)

    s, R, t = umeyama(est[:, :3, 3], gt[:, :3, 3])


    print(f"s: {s:.6f} (true {s_true})")
    print(f"R err: {np.abs(R - R_true).max():.2e}")
    print(f"t err: {np.abs(t - t_true).max():.2e}")
    print(f"det(R): {np.linalg.det(R):.6f}")

    assert np.isclose(s, s_true, atol=1e-6)
    assert np.allclose(R, R_true, atol=1e-6)
    assert np.allclose(t, t_true, atol=1e-4)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_ate_sim3_vs_se3():
    gt = load_poses(data_root_path / "poses/00.txt")
    est, *_ = make_distorted_est(gt)

    rmse_sim3, _ = ate(est, gt, align="sim3")
    rmse_se3, _  = ate(est, gt, align="se3")
    rmse_none, _  = ate(est, gt, align="none")

    

    print(f"ATE sim3: {rmse_sim3:.2e} m")
    print(f"ATE se3:  {rmse_se3:.2f} m")
    print(f"ATE none: {rmse_none:.2f} m")

    assert rmse_sim3 < 1e-6
    assert rmse_se3 > 1.0  # can't undo the 2.5x scale


def test_sim3_raises_on_zero_spread():
    # Zero-motion baseline: every est position is identical, so scale is undefined.
    gt = load_poses(data_root_path / "poses/00.txt")
    baseline = np.tile(np.eye(4), (gt.shape[0], 1, 1))

    try:
        ate(baseline, gt, align="sim3")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for zero-spread est")


def test_rpe():
    gt = load_poses(data_root_path / "poses/00.txt")
    est, *_ = make_distorted_est(gt)
    
    # Identical trajectories: no relative error at all.
    t_err, r_err = rpe(gt, gt)
    # assert t_err < 1e-9 and r_err < 1e-6

    # # Raw est has the 2.5x scale baked in, so translation RPE is large...
    # t_raw, r_raw = rpe(est, gt, delta=10)
    # # ...but a global rotation doesn't change relative rotations.
    # assert t_raw > 1.0
    # assert r_raw < 1e-6

    # # After Sim(3) alignment the scale is fixed and RPE goes to ~0.
    # _, aligned = ate(est, gt, align="sim3")
    # t_al, r_al = rpe(aligned, gt, delta=10)
    # print(f"RPE raw: {t_raw:.2f} m, aligned: {t_al:.2e} m")
    # assert t_al < 1e-6 and r_al < 1e-6


if __name__ == "__main__":
    # test_umeyama_recovers_known_transform()
    # test_ate_sim3_vs_se3()
    # test_sim3_raises_on_zero_spread()
    test_rpe()
    # print("all passed")
    
