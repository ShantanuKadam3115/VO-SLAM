"""Bundle adjustment for many stereo keyframes.

Same maths as p5_BALearning.py, but generalised from "2 frames, k0 fixed" to
"any number of poses, any number of points".

Data layout (an observation = one point seen in one camera):
    T_cw     (n_cams, 4, 4)  world -> camera poses
    X_w      (n_pts, 3)      world points
    cam_idx  (n_obs,)        which camera made observation i
    pt_idx   (n_obs,)        which point observation i is of
    obs      (n_obs, 3)      observed (u_L, u_R, v); u_R is NaN if there was no stereo match

So observation i says: camera cam_idx[i] saw point pt_idx[i] at pixel obs[i].
Nothing assumes every camera sees every point, which is what we need on KITTI.
"""
import cv2 as cv
import numpy as np


def skew(w):
    # [w]x for each row of w (N,3) -> (N,3,3), so that [w]x @ a = w x a
    S = np.zeros((len(w), 3, 3))
    S[:,0,1] = -w[:,2]
    S[:,0,2] = w[:,1]
    S[:,1,0] = w[:,2]
    S[:,1,2] = -w[:,0]
    S[:,2,0] = -w[:,1]
    S[:,2,1] = w[:,0]
    return S


def se3_exp(xi):
    # xi = (rho, phi) -> 4x4 transform
    rho = xi[:3]
    phi = xi[3:]
    theta = np.linalg.norm(phi)

    W = skew(phi.reshape(1, 3))[0]
    if theta < 1e-10:
        V = np.eye(3) + 0.5 * W
    else:
        V = np.eye(3) + (1 - np.cos(theta)) / theta**2 * W + (theta - np.sin(theta)) / theta**3 * W @ W

    T = np.eye(4)
    T[:3, :3] = cv.Rodrigues(phi)[0]
    T[:3, 3] = V @ rho
    return T


def transform_obs(T_cw, X_w, cam_idx, pt_idx):
    # X_cam for every observation: X_c = R @ X_w + t, with R, t of the camera that saw it
    # p5_BALearning had one T for all points; here each observation picks its own T
    R = T_cw[cam_idx, :3, :3]  # (n_obs, 3, 3)
    t = T_cw[cam_idx, :3, 3]   # (n_obs, 3)
    X = X_w[pt_idx]            # (n_obs, 3)

    X_cam = (R @ X[:, :, None])[:, :, 0] + t
    return X_cam


def project_stereo(X_cam, K, B):
    # predicted (u_L, u_R, v) for camera-frame points, rectified stereo pair
    f = K[0, 0]
    Cx = K[0, 2]
    Cy = K[1, 2]

    U_L_pred = (f * X_cam[:,0]) / X_cam[:,2] + Cx
    U_R_pred = (f * (X_cam[:,0] - B)) / X_cam[:,2] + Cx
    V_pred = (f * X_cam[:,1]) / X_cam[:,2] + Cy

    return np.column_stack((U_L_pred, U_R_pred, V_pred))


def residuals(T_cw, X_w, cam_idx, pt_idx, obs, K, B):
    # residual = observed - predicted, one row (u_L, u_R, v) per observation
    # rows with no u_R stay NaN in column 1; the solver will skip them
    X_cam = transform_obs(T_cw, X_w, cam_idx, pt_idx)
    pred = project_stereo(X_cam, K, B)
    return obs - pred


## ---- Jacobians: same 3 as p5_BALearning, now one block per observation ----

def projection_jacobian(X_cam, K, B):
    # Jacobian 1: d(u_L, u_R, v)/d(X_c, Y_c, Z_c), one 3x3 per observation
    #   u_L = f*X/Z + Cx      -> [f/Z, 0,   -f*X/Z^2    ]
    #   u_R = f*(X-B)/Z + Cx  -> [f/Z, 0,   -f*(X-B)/Z^2]
    #   v   = f*Y/Z + Cy      -> [0,   f/Z, -f*Y/Z^2    ]
    f = K[0, 0]
    Xc = X_cam[:,0]
    Yc = X_cam[:,1]
    Zc = X_cam[:,2]

    J = np.zeros((len(X_cam), 3, 3))

    J[:,0,0] = f / Zc
    J[:,0,2] = -f * Xc / Zc**2

    J[:,1,0] = f / Zc
    J[:,1,2] = -f * (Xc - B) / Zc**2

    J[:,2,1] = f / Zc
    J[:,2,2] = -f * Yc / Zc**2

    return J


def pose_jacobian(X_cam):
    # Jacobian 2: d(X_c)/d(xi) for the LEFT update T_new = exp(xi) @ T, one 3x6 per observation
    #   d/d(rho) = I,   d/d(phi) = -[X_c]x
    J = np.zeros((len(X_cam), 3, 6))
    J[:,:,:3] = np.eye(3)
    J[:,:,3:] = -skew(X_cam)
    return J


def obs_jacobians(T_cw, X_w, cam_idx, pt_idx, K, B):
    # chain rule, minus because res = obs - pred:
    #   d(res_i)/d(xi of cam_idx[i])  = -J_proj @ J_pose       (3x6)
    #   d(res_i)/d(X_w of pt_idx[i])  = -J_proj @ R            (3x3)  Jacobian 3: d(X_c)/d(X_w) = R
    # observation i only touches ONE pose and ONE point, so these two blocks are
    # all of its row in the big J; every other column of that row is 0
    X_cam = transform_obs(T_cw, X_w, cam_idx, pt_idx)
    J_proj = projection_jacobian(X_cam, K, B) # (n_obs, 3, 3)
    R = T_cw[cam_idx, :3, :3]                 # (n_obs, 3, 3)

    J_res_pose = -J_proj @ pose_jacobian(X_cam) # (n_obs, 3, 6)
    J_res_point = -J_proj @ R                   # (n_obs, 3, 3)

    return J_res_pose, J_res_point


## ---- Normal equations H dx = -g, built block by block (no big J) ----
#
# parameter vector x = [ xi_0 .. xi_{n_cams-1} | X_0 .. X_{n_pts-1} ]   (6 per camera, 3 per point)
#
#         | H_pp   H_pl |          | g_p |
#   H  =  |             |    g  =  |     |      H = J^T W J,  g = J^T W r
#         | H_lp   H_ll |          | g_l |
#
# observation i has non-zero J only in camera c = cam_idx[i] and point p = pt_idx[i], so it adds
#   Jc^T W Jc  to the 6x6 diagonal block of camera c      (H_pp)
#   Jp^T W Jp  to the 3x3 diagonal block of point p       (H_ll)
#   Jc^T W Jp  to the 6x3 block linking camera c, point p (H_pl, one block per observation)
# H_pp and H_ll are BLOCK DIAGONAL: two cameras never share an observation, nor do two points.
# Cameras are only coupled to each other through the points they both see (via H_pl).

def build_normal_equations(J_res_pose, J_res_point, res, w, cam_idx, pt_idx, n_cams, n_pts):
    # w: (n_obs, 3) weight per residual row (1 = plain least squares, Huber weights later)

    # rows with no u_R observation: weight 0 and residual 0, so they add nothing
    valid = np.isfinite(res)
    w = np.where(valid, w, 0.0)
    r = np.where(valid, res, 0.0)

    # W J: scale each residual row of the Jacobian by its weight
    WJ_pose = w[:, :, None] * J_res_pose   # (n_obs, 3, 6)
    WJ_point = w[:, :, None] * J_res_point # (n_obs, 3, 3)
    Wr = w * r                             # (n_obs, 3)

    # one contribution per observation
    H_pp_obs = J_res_pose.mT @ WJ_pose            # (n_obs, 6, 6)
    H_ll_obs = J_res_point.mT @ WJ_point          # (n_obs, 3, 3)
    H_pl = J_res_pose.mT @ WJ_point               # (n_obs, 6, 3)
    g_p_obs = (J_res_pose.mT @ Wr[:, :, None])[:, :, 0]   # (n_obs, 6)
    g_l_obs = (J_res_point.mT @ Wr[:, :, None])[:, :, 0]  # (n_obs, 3)

    # sum the contributions of all observations of the same camera / point
    # np.add.at instead of H_pp[cam_idx] += ...: plain += keeps only ONE of the repeated
    # indices (camera 0 appears in hundreds of observations), add.at adds every one
    H_pp = np.zeros((n_cams, 6, 6))
    H_ll = np.zeros((n_pts, 3, 3))
    g_p = np.zeros((n_cams, 6))
    g_l = np.zeros((n_pts, 3))
    np.add.at(H_pp, cam_idx, H_pp_obs)
    np.add.at(H_ll, pt_idx, H_ll_obs)
    np.add.at(g_p, cam_idx, g_p_obs)
    np.add.at(g_l, pt_idx, g_l_obs)

    return H_pp, H_pl, H_ll, g_p, g_l
