from vslam.kitti_io import load_poses
import numpy as np
from vslam.config import data_root_path
import glob
import os

calib_path = data_root_path / "sequences/00/calib.txt"
pose_path = data_root_path / "poses/00.txt"
image_dir_path = data_root_path / "sequences/00/image_0"

image_files_path = os.path.join(image_dir_path, "*.png")
num_frames = len(glob.glob(image_files_path))

poses = load_poses(pose_path)
R = poses[:, :3, :3]                                     
t = poses[:, :3, 3] 

consecutive_distance = np.diff(t, axis=0)
length_of_each_step = np.linalg.norm(consecutive_distance, axis=1)
total_distance = length_of_each_step.sum()

# print(np.linalg.det(R))
R_t = R.transpose(0,2,1)
# print(R_t.shape, R.shape)
# print(np.allclose(R_t@R, np.eye(3)))

mask = np.isclose(np.linalg.det(R), 1.0, atol=1e-6)

# print(mask.all())
# print((~mask).sum())

print(poses.shape)
print(poses[0])
print(t[0])

# print(np.allclose(poses[0],np.eye(4)))
# print(num_frames== len(poses))
# print(np.diff(t,axis=0).shape)
# print(total_distance)
# print(np.linalg.norm(t[-1] - t[0]))
# print(np.linalg.norm(t[0] - t[1]))

