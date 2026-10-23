import numpy as np


def triangulate(P1, P2, pts1, pts2):
    """Linear (DLT) triangulation of matched points.

    P1, P2: (3, 4) projection matrices.
    pts1, pts2: (N, 2) pixel coordinates of the matches.
    Returns X: (N, 3) points in the world frame of P1/P2.

    x ~ P X  <=>  x × (P X) = 0. With x = (u, v, 1) and p_i the rows of P,
    the cross product is
        [ v (p3·X) - (p2·X),
          (p1·X) - u (p3·X),
          u (p2·X) - v (p1·X) ]
    The third row is a combination of the first two, so each camera gives
    2 equations: u p3 - p1 and v p3 - p2 (signs don't matter, = 0).
    """
    P1 = np.asarray(P1, dtype=float)
    P2 = np.asarray(P2, dtype=float)
    pts1 = np.asarray(pts1, dtype=float)
    pts2 = np.asarray(pts2, dtype=float)
    assert pts1.shape == pts2.shape and pts1.shape[1] == 2

    u1, v1 = pts1[:, 0:1], pts1[:, 1:2]  # (N, 1) so they broadcast against (4,) rows
    u2, v2 = pts2[:, 0:1], pts2[:, 1:2]


    # (N, 4, 4): one A per match, rows stacked along axis 1
    A = np.stack([
        u1 * P1[2] - P1[0],
        v1 * P1[2] - P1[1],
        u2 * P2[2] - P2[0],
        v2 * P2[2] - P2[1],
    ], axis=1)


    # A X = 0: X is the right singular vector for the smallest singular value
    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[:, -1, :]  # (N, 4)

    return X_h[:, :3] / X_h[:, 3:4]


def project(P, X):
    """Project (N, 3) points with a (3, 4) camera matrix -> (N, 2) pixels."""
    X_h = np.hstack([X, np.ones((len(X), 1))])
    x = X_h @ P.T
    return x[:, :2] / x[:, 2:3]
