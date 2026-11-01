import cv2
import numpy as np


def make_stereo_matcher(method="sgbm", num_disparities=128, block_size=None):
    """Dense stereo matcher for rectified pairs.

    method: "bm" (plain block matching: fast, holes in low texture) or
            "sgbm" (semi-global: block matching + smoothness along 8 paths).
    num_disparities: search range in px, must be a multiple of 16. With KITTI's
            f = 718 px and B = 0.54 m, 128 px covers depths down to ~3 m.
    """
    if method == "bm":
        return cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size or 15)
    if method == "sgbm":
        b = block_size or 5
        return cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=num_disparities, blockSize=b,
            P1=8 * b * b, P2=32 * b * b,      # penalties for disparity changes of 1 / more than 1
            disp12MaxDiff=1, uniquenessRatio=10,
            speckleWindowSize=100, speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    raise ValueError(f"unknown stereo method: {method!r}")


def compute_disparity(left, right, matcher):
    """Disparity in pixels (float32). Invalid pixels come out <= 0."""
    # OpenCV returns fixed-point int16 with 4 fractional bits
    return matcher.compute(left, right).astype(np.float32) / 16.0


def backproject(pts, disp, K, baseline, min_disp=1.0, max_depth=np.inf):
    """Metric 3D points (camera frame) for pixels with a valid disparity.

    pts: (N, 2) pixel coordinates. disp: (H, W) disparity map.
    Depth from disparity: z = f * B / d. Then x = (u - cx) z / fx, y = (v - cy) z / fy.
    Returns (X (N, 3), valid (N,) bool); invalid rows of X are NaN.
    """
    H, W = disp.shape
    u = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1)
    v = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
    d = disp[v, u]

    valid = d > min_disp
    z = np.full(len(pts), np.nan)
    z[valid] = K[0, 0] * baseline / d[valid]
    valid &= z < max_depth

    X = np.full((len(pts), 3), np.nan)
    X[valid, 0] = (pts[valid, 0] - K[0, 2]) * z[valid] / K[0, 0]
    X[valid, 1] = (pts[valid, 1] - K[1, 2]) * z[valid] / K[1, 1]
    X[valid, 2] = z[valid]
    return X, valid
