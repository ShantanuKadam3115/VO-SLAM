import numpy as np


def match_features(img1, img2, detector, matcher, ratio=0.75):
    """Detect, describe and match features between two images.

    detector: e.g. cv2.ORB_create(...), created once by the caller.
    matcher:  e.g. cv2.BFMatcher(cv2.NORM_HAMMING), without crossCheck.
    ratio:    Lowe's ratio-test threshold.
    Returns (pts1, pts2): (N, 2) float32 pixel coordinates of matched points,
    ready for cv2.findEssentialMat.
    """
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    # A featureless frame gives des = None; knnMatch would crash on it
    if des1 is None or des2 is None:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)

    matches = matcher.knnMatch(des1, des2, k=2)

    # Ratio test (a point can come back with < 2 neighbours, so skip those)
    good = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 2)
    return pts1, pts2
