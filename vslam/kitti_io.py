import numpy as np
import cv2




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


def load_image(seq_dir, idx, cam=0):
    # e.g. D:\CV\Data\kitti\dataset\sequences\00\image_0\000000.png
    image_path = seq_dir / f"image_{cam}" / f"{idx:06d}.png"

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"could not read image: {image_path}")
    return img


