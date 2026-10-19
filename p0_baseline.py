import numpy as np

from vslam.config import data_root_path
from vslam.kitti_io import load_poses
from vslam.metrics import ate, rpe

pose_path = data_root_path / "poses/00.txt"

gt = load_poses(pose_path)
N = gt.shape[0]

# "Never moves" baseline: the camera sits at the origin for every frame.
baseline = np.tile(np.eye(4), (N, 1, 1))


rmse_none, _ = ate(baseline, gt, align="none")
rmse_se3, aligned_se3 = ate(baseline, gt, align="se3")
# delta=1: per-frame drift (KITTI is 10 Hz, so 0.1 s per step)
rpe_t, rpe_r = rpe(aligned_se3, gt, delta=1)

# No Sim(3): all baseline positions are identical, so scale is undefined (umeyama raises).
print(f"Zero-motion baseline on KITTI 00 ({N} frames)")
print(f"  ATE (none): {rmse_none:.2f} m")
print(f"  ATE (se3):  {rmse_se3:.2f} m")
print(f"  RPE (d=1):  {rpe_t:.3f} m, {rpe_r:.3f} deg")
