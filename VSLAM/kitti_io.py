import numpy as np
import matplotlib.pyplot as plt
from config import data_root_path


# calib_path = data_root_path / "sequences/00/calib.txt"
pose_path = data_root_path / "poses/00.txt"

def load_calib(path):
    calib = {}
    for line in open(path , "r", encoding="utf-8"):
        key, vals = line.split(":", 1)
        calib[key] = np.array(vals.split(), dtype=float).reshape(3, 4)

    P0 = calib["P0"]
    K  = P0[:, :3]
    B  = -calib["P1"][0, 3] / calib["P1"][0, 0]   # 0.537 m
    return K, B


def load_poses(path):
    poses_34 = np.loadtxt(path).reshape(-1, 3, 4)
    N = poses_34.shape[0]
    identity_stack = np.tile(np.eye(4),(N,1,1))
    identity_stack[:, :3, :] = poses_34

    # R = poses[:, :, :3]                                     
    # t = poses[:, :, 3] 
    return identity_stack                                     

# check = load_poses(pose_path)
# print(check.shape)
# print(check[0])
# print(calib_path)
# K,B = load_calib(calib_path)
# print(K)
# plt.plot(t[:, 0], t[:, 2])          # x vs z = bird's-eye view
# plt.axis("equal"); plt.xlabel("x (m)"); plt.ylabel("z (m)")
# plt.show()



