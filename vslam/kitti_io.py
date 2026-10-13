import numpy as np




def load_calib(path):
    calib = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
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
    return identity_stack                                     



