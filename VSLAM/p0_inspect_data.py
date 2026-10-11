from VSLAM.kitti_io import load_calib, load_poses
import numpy as np
from config import data_root_path

calib_path = data_root_path / "sequences/00/calib.txt"
pose_path = data_root_path / "poses/00.txt"

poses = load_poses(pose_path)
R = poses[:, :3, :3]                                     
t = poses[:, :3, 3] 


print(poses.shape)
# print(poses[0])
print(R[0])

print(np.allclose(R[0],np.eye(3)))

