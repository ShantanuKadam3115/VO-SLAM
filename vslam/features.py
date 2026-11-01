import numpy as np


def match_features(img1, img2, detector, matcher, ratio=0.75):
    """Detect, describe and match features between two images.

    detector: e.g. cv2.ORB_create(...), created once by the caller.
    matcher:  e.g. cv2.BFMatcher(cv2.NORM_HAMMING), without crossCheck.
    ratio:    Lowe's ratio-test threshold.
    Returns (pts1, pts2): (N, 2) float32 pixel coordinates of matched points,
    ready for cv2.findEssentialMat.
    """
    pts1, des1 = detect(img1, detector)
    pts2, des2 = detect(img2, detector)
    idx1, idx2 = match_descriptors(des1, des2, matcher, ratio)
    return pts1[idx1], pts2[idx2]


def detect(img, detector):
    """Detect and describe features -> (pts (N, 2) float32, des (N, D) or None)."""
    kp, des = detector.detectAndCompute(img, None)
    pts = np.float32([k.pt for k in kp]).reshape(-1, 2)
    return pts, des


def match_by_projection(proj, map_des, pts, des, matcher, radius=15.0, max_dist=64, ratio=0.9):
    """Match map points to features near where the map points are predicted to appear.

    proj: (M, 2) predicted pixel positions of the map points.
    map_des: (M, D) map descriptors. pts, des: the frame's features.
    Only features within `radius` px are candidates, so the ratio test only has
    to beat nearby features (ORB-SLAM uses ~0.9 here instead of 0.75).
    Returns (map_idx, feat_idx), one-to-one.
    """
    if des is None or len(map_des) == 0 or len(des) == 0:
        return np.empty(0, int), np.empty(0, int)

    # (M, F) mask of allowed pairs; uint8 as BFMatcher expects
    d2 = np.sum((proj[:, None, :] - pts[None, :, :]) ** 2, axis=2)
    mask = (d2 < radius ** 2).astype(np.uint8)

    matches = matcher.knnMatch(map_des, des, k=2, mask=mask)

    best = {}  # feature idx -> (distance, map idx): keep one map point per feature
    for pair in matches:
        if len(pair) == 0:
            continue
        m = pair[0]
        if m.distance > max_dist:
            continue
        if len(pair) == 2 and m.distance >= ratio * pair[1].distance:
            continue
        if m.trainIdx not in best or m.distance < best[m.trainIdx][0]:
            best[m.trainIdx] = (m.distance, m.queryIdx)

    feat_idx = np.array(list(best.keys()), dtype=int)
    map_idx = np.array([v[1] for v in best.values()], dtype=int)
    return map_idx, feat_idx


def match_descriptors(des1, des2, matcher, ratio=0.75):
    """kNN matching + Lowe's ratio test on precomputed descriptors.

    Returns (idx1, idx2): int arrays, des1[idx1[k]] matches des2[idx2[k]].
    """
    # A featureless frame gives des = None; knnMatch would crash on it
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return np.empty(0, int), np.empty(0, int)

    matches = matcher.knnMatch(des1, des2, k=2)

    # Ratio test (a point can come back with < 2 neighbours, so skip those)
    good = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)

    idx1 = np.array([m.queryIdx for m in good], dtype=int)
    idx2 = np.array([m.trainIdx for m in good], dtype=int)
    return idx1, idx2
