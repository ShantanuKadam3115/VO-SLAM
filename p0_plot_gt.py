import matplotlib.pyplot as plt

from vslam.config import data_root_path
from vslam.kitti_io import load_poses
from vslam.plotting import plot_trajectories

pose_path = data_root_path / "poses/00.txt"

gt = load_poses(pose_path)

plot_trajectories([gt], ["ground truth"], "KITTI 00 - ground truth",
                  save_path="results/p0_gt_00.png")
plt.show()
