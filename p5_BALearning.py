import cv2 as cv
import numpy as np

from vslam.config import data_root_path
from vslam.kitti_io import load_calib, load_poses, load_image
from vslam.features import detect, match_descriptors

seq = "00"
seq_dir = data_root_path / "sequences" / seq
gt = load_poses(data_root_path / f"poses/{seq}.txt")
K, B = load_calib(seq_dir / "calib.txt")
N = len(gt)


f = K[0][0]
Cx = K[0][2]
Cy = K[1][2]


img_K0_L = load_image(seq_dir, 0, 0)
img_K0_R = load_image(seq_dir, 0, 1)


# feature matching
orb = cv.ORB_create(nfeatures=3000)
bf = cv.BFMatcher(cv.NORM_HAMMING)

pts_K0_L, des_K0_L = detect(img_K0_L, orb)
pts_K0_R, des_K0_R = detect(img_K0_R, orb)

idx_L, idx_R = match_descriptors(des_K0_L, des_K0_R, bf)

pts1 = pts_K0_L[idx_L]
pts2 = pts_K0_R[idx_R]


# stereo triangulation.

U_K0_L = pts1[:,0]
V_K0_L = pts1[:,1]

U_K0_R = pts2[:,0]
V_K0_R = pts2[:,1]

initial_disparity = U_K0_L - U_K0_R

d_min = 1
d_max = (f*B)/ 2


row_mask = np.abs(V_K0_L - V_K0_R) < 2
range_mask = (initial_disparity < d_max) & (initial_disparity > d_min)
valid = row_mask & range_mask

disparity = initial_disparity[valid]
u = U_K0_L[valid]
v = V_K0_L[valid]

Z = (f*B)/disparity
X = (u - Cx) * Z / f
Y = (v - Cy) * Z / f

World_points = np.column_stack((X,Y,Z))

# descriptor of each 3D point = descriptor of its K0 left keypoint
World_des = des_K0_L[idx_L][valid]

# Matching at frame k1(temporal match)

img_K1_L = load_image(seq_dir, 1, 0)
img_K1_R = load_image(seq_dir, 1, 1)

pts_K1_L, des_K1_L = detect(img_K1_L, orb)
pts_K1_R, des_K1_R = detect(img_K1_R, orb)

# stereo match at k1, only needed to get the observed u_R at k1
idx_K1_L_st, idx_K1_R_st = match_descriptors(des_K1_L, des_K1_R, bf)

disparity_K1 = pts_K1_L[idx_K1_L_st, 0] - pts_K1_R[idx_K1_R_st, 0]
row_mask_K1 = np.abs(pts_K1_L[idx_K1_L_st, 1] - pts_K1_R[idx_K1_R_st, 1]) < 2
range_mask_K1 = (disparity_K1 < d_max) & (disparity_K1 > d_min)
valid_K1 = row_mask_K1 & range_mask_K1


# u_R for every k1 left keypoint, NaN where it has no valid right match
U_K1_R_of_kp = np.full(len(pts_K1_L), np.nan)
U_K1_R_of_kp[idx_K1_L_st[valid_K1]] = pts_K1_R[idx_K1_R_st[valid_K1], 0]

# total_nans = np.sum(np.isnan(U_K1_R_of_kp))
# print(f"Total NaNs: {total_nans}")

# print(U_K1_R_of_kp.shape)

idx_W, idx_K1 = match_descriptors(World_des, des_K1_L, bf)

matched_World_points = World_points[idx_W] #448 world points.
matched_pts_K1_L = pts_K1_L[idx_K1]



#ransac for getting R,t

ok, rvec, tvec, inl = cv.solvePnPRansac(matched_World_points, matched_pts_K1_L, K, None, iterationsCount=200, reprojectionError= 3.0, confidence=0.999)

T_cw = np.eye(4)
T_cw[:3, :3] = cv.Rodrigues(rvec)[0]
T_cw[:3, 3] = tvec.ravel()

#predict and compute the residuals
R = T_cw[:3, :3]
t = T_cw[:3, 3]

# at frame k0

X_cam_k0 = matched_World_points # as world origin and cam k0 origin are same

U_K0_L_pred = (f * X_cam_k0[:,0])/X_cam_k0[:,2] + Cx
U_K0_R_pred = (f * (X_cam_k0[:,0]-B))/X_cam_k0[:,2] + Cx
V_K0_pred = (f * X_cam_k0[:,1])/X_cam_k0[:,2] + Cy # rectified: same v in left and right

# at frame k1

X_cam_k1 = (R@ matched_World_points.mT + t.reshape(3, 1)).mT

U_K1_L_pred = (f * X_cam_k1[:,0])/X_cam_k1[:,2] + Cx
U_K1_R_pred = (f * (X_cam_k1[:,0]-B))/X_cam_k1[:,2] + Cx
V_K1_pred = (f * X_cam_k1[:,1])/X_cam_k1[:,2] + Cy

# observed pixels, in the same row order as matched_World_points

U_K0_L_obs = u[idx_W]
U_K0_R_obs = U_K0_R[valid][idx_W]
V_K0_obs = v[idx_W]

U_K1_L_obs = matched_pts_K1_L[:,0]
U_K1_R_obs = U_K1_R_of_kp[idx_K1] # NaN where k1 had no stereo match
V_K1_obs = matched_pts_K1_L[:,1]

# total_nans = np.sum(np.isnan(U_K1_R_obs))
# print(f"Total NaNs: {total_nans}")

# residual = observed - predicted, one row per point: (u_L, u_R, v)

res_K0 = np.column_stack((U_K0_L_obs - U_K0_L_pred, U_K0_R_obs - U_K0_R_pred, V_K0_obs - V_K0_pred))
res_K1 = np.column_stack((U_K1_L_obs - U_K1_L_pred, U_K1_R_obs - U_K1_R_pred, V_K1_obs - V_K1_pred))


# only the RANSAC inliers go to BA, outliers would dominate the cost
inl = inl.ravel()

res_K0_inl = res_K0[inl]
res_K1_inl = res_K1[inl]

has_R_K1 = np.isfinite(res_K1_inl[:,1])
# print(res_K0_inl.shape, res_K1_inl.shape, has_R_K1.shape)
# print(np.sum(has_R_K1))

# Jacobian 1: projection, d(u_L, u_R, v)/d(X_c, Y_c, Z_c), one 3x3 per point
# rows follow the residual columns (u_L, u_R, v)
#   u_L = f*X/Z + Cx      -> [f/Z, 0,   -f*X/Z^2    ]
#   u_R = f*(X-B)/Z + Cx  -> [f/Z, 0,   -f*(X-B)/Z^2]
#   v   = f*Y/Z + Cy      -> [0,   f/Z, -f*Y/Z^2    ]

def projection_jacobian(X_cam):
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

J_proj_K0 = projection_jacobian(X_cam_k0[inl]) # (420, 3, 3)
J_proj_K1 = projection_jacobian(X_cam_k1[inl])

# residual = obs - pred, so d(res)/d(X_cam) = -J_proj. The sign is applied when Jacobians get chained.

# Jacobian 2: pose, d(X_c)/d(xi), one 3x6 per point, only for k1 (k0 is fixed, it defines the world)
# xi = (rho, phi): 3 translation + 3 rotation, applied on the LEFT of the pose:
#   T_cw_new = exp(xi) @ T_cw
# for a small xi:  X_c_new = X_c + phi x X_c + rho
#   d/d(rho) = I
#   d/d(phi) = -[X_c]x      (phi x X_c = -X_c x phi = -[X_c]x phi)
# the same left-update rule must be used later when BA applies its step

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

def pose_jacobian(X_cam):
    J = np.zeros((len(X_cam), 3, 6))
    J[:,:,:3] = np.eye(3)
    J[:,:,3:] = -skew(X_cam)
    return J

J_pose_K1 = pose_jacobian(X_cam_k1[inl]) # (420, 3, 6)

# Jacobian 3: point, d(X_c)/d(X_w), one 3x3 per point and frame
#   X_c = R @ X_w + t  ->  d(X_c)/d(X_w) = R   (same for every point)
#   at k0, T_cw = I    ->  d(X_c)/d(X_w) = I

J_point_K0 = np.tile(np.eye(3), (len(inl), 1, 1)) # (420, 3, 3)
J_point_K1 = np.tile(R, (len(inl), 1, 1))

# chain rule: residual Jacobians that BA uses, minus because res = obs - pred
#   d(res_K1)/d(xi)  = -J_proj_K1 @ J_pose_K1    (3x6 per point)
#   d(res_K0)/d(X_w) = -J_proj_K0 @ J_point_K0   (3x3 per point)
#   d(res_K1)/d(X_w) = -J_proj_K1 @ J_point_K1   (3x3 per point)

J_res_K1_pose = -J_proj_K1 @ J_pose_K1 # (420, 3, 6)
J_res_K0_point = -J_proj_K0 @ J_point_K0 # (420, 3, 3)
J_res_K1_point = -J_proj_K1 @ J_point_K1 # (420, 3, 3)



n_pts = len(inl)
n_res = 2 * 3 * n_pts # 2520
n_par = 6 + 3 * n_pts # 1266

J_global = np.zeros((n_res, n_par))

for i in range(n_pts):
    row_K0 = 3*i
    row_K1 = 3*n_pts + 3*i
    col_pt = 6 + 3*i

    J_global[row_K0:row_K0+3, col_pt:col_pt+3] = J_res_K0_point[i]

    J_global[row_K1:row_K1+3, 0:6] = J_res_K1_pose[i]
    J_global[row_K1:row_K1+3, col_pt:col_pt+3] = J_res_K1_point[i]

# residual vector in the same row order
res_global = np.concatenate((res_K0_inl.ravel(), res_K1_inl.ravel())) # (2520,)

# drop rows where k1 had no u_R observation (NaN residual)
valid_rows = np.isfinite(res_global)

J_global = J_global[valid_rows]
res_global = res_global[valid_rows]

# print(J_global.shape, res_global.shape)

# Levenberg-Marquardt loop
# each iteration: linearise at the current estimate -> solve for delta_x -> try the step
#   cost went down -> accept, trust the linear model more (damp / 10)
#   cost went up   -> reject, take smaller, more gradient-like steps (damp * 10)

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

def transform(T, X_w):
    return (T[:3, :3] @ X_w.mT + T[:3, 3].reshape(3, 1)).mT

def stereo_residuals(T, X_w, U_L_obs, U_R_obs, V_obs):
    X_cam = transform(T, X_w)

    U_L_pred = (f * X_cam[:,0])/X_cam[:,2] + Cx
    U_R_pred = (f * (X_cam[:,0]-B))/X_cam[:,2] + Cx
    V_pred = (f * X_cam[:,1])/X_cam[:,2] + Cy

    return np.column_stack((U_L_obs - U_L_pred, U_R_obs - U_R_pred, V_obs - V_pred))

def stack_residuals(T, X_w):
    # same row order and same NaN rows as res_global
    res_K0 = stereo_residuals(np.eye(4), X_w, U_K0_L_obs[inl], U_K0_R_obs[inl], V_K0_obs[inl])
    res_K1 = stereo_residuals(T, X_w, U_K1_L_obs[inl], U_K1_R_obs[inl], V_K1_obs[inl])
    return np.concatenate((res_K0.ravel(), res_K1.ravel()))[valid_rows]

def build_J_global(T, X_w):
    # same steps as above (Jacobians 1-3 + chain rule + placement), at the current estimate
    X_cam_0 = X_w
    X_cam_1 = transform(T, X_w)

    J_res_K1_pose = -projection_jacobian(X_cam_1) @ pose_jacobian(X_cam_1)
    J_res_K0_point = -projection_jacobian(X_cam_0) # J_point_K0 = I
    J_res_K1_point = -projection_jacobian(X_cam_1) @ T[:3, :3] # J_point_K1 = R

    J = np.zeros((n_res, n_par))
    for i in range(n_pts):
        row_K0 = 3*i
        row_K1 = 3*n_pts + 3*i
        col_pt = 6 + 3*i

        J[row_K0:row_K0+3, col_pt:col_pt+3] = J_res_K0_point[i]

        J[row_K1:row_K1+3, 0:6] = J_res_K1_pose[i]
        J[row_K1:row_K1+3, col_pt:col_pt+3] = J_res_K1_point[i]

    return J[valid_rows]

T_est = T_cw.copy()
X_est = matched_World_points[inl].copy()

damp = 1e-3
max_iter = 20

res = stack_residuals(T_est, X_est)
cost = 0.5 * res @ res
J = build_J_global(T_est, X_est)

print(f"iter  0  cost {cost:10.3f}  RMS {np.sqrt(np.mean(res**2)):.4f} px")

for it in range(1, max_iter + 1):
    H = J.mT @ J
    g = J.mT @ res

    delta_x = np.linalg.solve(H + damp * np.eye(n_par), -g)

    # try the step: LEFT update for the pose (same rule as Jacobian 2), plain add for points
    T_try = se3_exp(delta_x[:6]) @ T_est
    X_try = X_est + delta_x[6:].reshape(n_pts, 3)

    res_try = stack_residuals(T_try, X_try)
    cost_try = 0.5 * res_try @ res_try

    if cost_try < cost:
        cost_drop = cost - cost_try

        T_est, X_est, res, cost = T_try, X_try, res_try, cost_try
        J = build_J_global(T_est, X_est) # re-linearise only after an accepted step
        damp /= 10

        print(f"iter {it:2d}  cost {cost:10.3f}  RMS {np.sqrt(np.mean(res**2)):.4f} px  |dx| {np.linalg.norm(delta_x):.2e}  damp {damp:.0e}  accepted")

        if cost_drop < 1e-6 * cost or np.linalg.norm(delta_x) < 1e-8:
            print("converged")
            break
    else:
        damp *= 10
        print(f"iter {it:2d}  cost {cost_try:10.3f}  rejected, damp -> {damp:.0e}")

C_pnp = -T_cw[:3, :3].T @ T_cw[:3, 3]
C_ba = -T_est[:3, :3].T @ T_est[:3, 3]
print("k1 camera centre  PnP:", C_pnp, "  BA:", C_ba)








