import matplotlib.pyplot as plt

from vslam.config import RESULTS_DIR, data_root_path
from vslam.kitti_io import load_poses
from vslam.plotting import plot_trajectories

pose_path = data_root_path / "poses/00.txt"

gt = load_poses(pose_path)

plot_trajectories([gt], ["ground truth"], "KITTI 00 - ground truth",
                  save_path=RESULTS_DIR / "p0_gt_00.png")
plt.show()
