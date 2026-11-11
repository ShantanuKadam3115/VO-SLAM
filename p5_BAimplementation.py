"""Phase 5: multi-keyframe BA. Step 1: test vslam/ba.py on a synthetic scene.

Synthetic = we make up the true poses and points ourselves, so we know the right
answer exactly. Once BA recovers it here, we plug it into KITTI.
"""
import cv2 as cv
import numpy as np

from vslam.config import data_root_path
from vslam.kitti_io import load_calib
from vslam.ba import residuals, transform_obs, project_stereo, obs_jacobians, se3_exp, build_normal_equations

rng = np.random.default_rng(0)

## real KITTI intrinsics + baseline, so the pixel numbers look like KITTI
seq_dir = data_root_path / "sequences" / "00"
K, B = load_calib(seq_dir / "calib.txt")
W_img, H_img = 1241, 376


## ---- True cameras: a car driving forward 1 m per frame, turning slightly ----
n_cams = 5

T_wc_true = np.tile(np.eye(4), (n_cams, 1, 1)) # camera -> world (like est in p4)

for k in range(n_cams):
    yaw = np.deg2rad(2.0 * k) # 2 deg per frame around the camera y axis (KITTI y points down)
    T_wc_true[k, :3, :3] = cv.Rodrigues(np.array([0.0, yaw, 0.0]))[0]
    T_wc_true[k, :3, 3] = [0.0, 0.0, 1.0 * k]

T_cw_true = np.linalg.inv(T_wc_true) # BA works with world -> camera


## ---- True points: a box in front of the cars, like KITTI's 2-40 m range ----
n_pts = 300
X_w_true = np.column_stack((rng.uniform(-15, 15, n_pts),   # left/right
                            rng.uniform(-3, 2, n_pts),     # up/down (road is ~ +1.7 m)
                            rng.uniform(10, 40, n_pts)))   # forward


## ---- Observations: every (camera, point) pair that is actually visible ----
cam_idx = []
pt_idx = []
for k in range(n_cams):
    for i in range(n_pts):
        cam_idx.append(k)
        pt_idx.append(i)
cam_idx = np.array(cam_idx)
pt_idx = np.array(pt_idx)


X_cam = transform_obs(T_cw_true, X_w_true, cam_idx, pt_idx)
pix = project_stereo(X_cam, K, B) #predicted u_l, u_r, v_l


# visible = in front of the camera and inside the left image
visible = (X_cam[:,2] > 1.0) & (pix[:,0] >= 0) & (pix[:,0] < W_img) & (pix[:,2] >= 0) & (pix[:,2] < H_img)
cam_idx = cam_idx[visible]
pt_idx = pt_idx[visible]
obs_true = pix[visible]

print(f"{n_cams} cameras, {n_pts} points, {len(cam_idx)} observations "
      f"(out of {n_cams * n_pts} possible)")


## ---- Check 1: at the true poses/points, with perfect pixels, residuals are 0 ----
res = residuals(T_cw_true, X_w_true, cam_idx, pt_idx, obs_true, K, B)
print(f"check 1, true state + perfect pixels: max |res| = {np.max(np.abs(res)):.2e} px  (expect ~1e-12)")


## ---- Check 2: noisy pixels -> RMS residual ~ the noise we added ----
pix_noise = 0.5 # px
obs = obs_true + rng.normal(0, pix_noise, obs_true.shape)

# like KITTI: some observations have no right-image match -> u_R unknown
no_right = rng.random(len(obs)) < 0.2
obs[no_right, 1] = np.nan

res = residuals(T_cw_true, X_w_true, cam_idx, pt_idx, obs, K, B)
print(f"check 2, true state + {pix_noise} px noise: RMS = {np.sqrt(np.nanmean(res**2)):.3f} px  "
      f"(expect ~{pix_noise}), NaN u_R rows = {np.sum(np.isnan(res[:,1]))}")


## ---- Check 3: analytic Jacobians vs finite differences ----
# numerical derivative: nudge one parameter by +eps and -eps, see how the residual moves
#   d(res)/d(param) ~ (res(param + eps) - res(param - eps)) / (2 eps)
# if this matches obs_jacobians, the derivation and the code are both right

# test away from the true state, so we don't only check one lucky point
T_cw_test = T_cw_true.copy()
for k in range(n_cams):
    T_cw_test[k] = se3_exp(rng.normal(0, 0.05, 6)) @ T_cw_true[k]
X_w_test = X_w_true + rng.normal(0, 0.3, X_w_true.shape)

J_res_pose, J_res_point = obs_jacobians(T_cw_test, X_w_test, cam_idx, pt_idx, K, B)

eps = 1e-6

# pose: observation i only depends on camera cam_idx[i], so we can nudge ALL cameras
# along direction j at the same time and still read off each observation's own column j
J_pose_num = np.zeros_like(J_res_pose)
for j in range(6):
    d = np.zeros(6)
    d[j] = eps

    T_plus = T_cw_test.copy()
    T_minus = T_cw_test.copy()
    for k in range(n_cams):
        T_plus[k] = se3_exp(d) @ T_cw_test[k]   # same LEFT update as the Jacobian assumes
        T_minus[k] = se3_exp(-d) @ T_cw_test[k]

    res_plus = residuals(T_plus, X_w_test, cam_idx, pt_idx, obs_true, K, B)
    res_minus = residuals(T_minus, X_w_test, cam_idx, pt_idx, obs_true, K, B)
    J_pose_num[:, :, j] = (res_plus - res_minus) / (2 * eps)

# point: same trick, observation i only depends on point pt_idx[i]
J_point_num = np.zeros_like(J_res_point)
for j in range(3):
    X_plus = X_w_test.copy()
    X_minus = X_w_test.copy()
    X_plus[:, j] += eps
    X_minus[:, j] -= eps

    res_plus = residuals(T_cw_test, X_plus, cam_idx, pt_idx, obs_true, K, B)
    res_minus = residuals(T_cw_test, X_minus, cam_idx, pt_idx, obs_true, K, B)
    J_point_num[:, :, j] = (res_plus - res_minus) / (2 * eps)

# relative error, since the entries range from ~1 to ~1000 px per unit
err_pose = np.max(np.abs(J_res_pose - J_pose_num)) / np.max(np.abs(J_res_pose))
err_point = np.max(np.abs(J_res_point - J_point_num)) / np.max(np.abs(J_res_point))
print(f"check 3, Jacobian vs finite diff: pose rel err = {err_pose:.1e}, point rel err = {err_point:.1e}  (expect < 1e-6)")


## ---- Check 4: block normal equations vs the dense J^T W J ----
# use the noisy obs (with NaN u_R rows) and random weights, so the NaN and weight paths get tested
res_test = residuals(T_cw_test, X_w_test, cam_idx, pt_idx, obs, K, B)
w_test = rng.uniform(0.2, 1.0, res_test.shape)

H_pp, H_pl, H_ll, g_p, g_l = build_normal_equations(J_res_pose, J_res_point, res_test, w_test,
                                                    cam_idx, pt_idx, n_cams, n_pts)

# the old way (like build_J_global in p5_BALearning): one big J, observation i -> rows 3i..3i+2
n_obs = len(cam_idx)
n_par = 6 * n_cams + 3 * n_pts
J_dense = np.zeros((3 * n_obs, n_par))
for i in range(n_obs):
    row = 3 * i
    col_cam = 6 * cam_idx[i]
    col_pt = 6 * n_cams + 3 * pt_idx[i]
    J_dense[row:row+3, col_cam:col_cam+6] = J_res_pose[i]
    J_dense[row:row+3, col_pt:col_pt+3] = J_res_point[i]

r_dense = res_test.ravel()
w_dense = w_test.ravel()
valid_rows = np.isfinite(r_dense)
J_dense = J_dense[valid_rows]
r_dense = r_dense[valid_rows]
w_dense = w_dense[valid_rows]

H_dense = J_dense.T @ (w_dense[:, None] * J_dense)
g_dense = J_dense.T @ (w_dense * r_dense)

# put our blocks into a full matrix with the same layout
H_blocks = np.zeros((n_par, n_par))
g_blocks = np.zeros(n_par)
for k in range(n_cams):
    H_blocks[6*k:6*k+6, 6*k:6*k+6] = H_pp[k]
    g_blocks[6*k:6*k+6] = g_p[k]
for p in range(n_pts):
    c = 6 * n_cams + 3 * p
    H_blocks[c:c+3, c:c+3] = H_ll[p]
    g_blocks[c:c+3] = g_l[p]
for i in range(n_obs):
    col_cam = 6 * cam_idx[i]
    col_pt = 6 * n_cams + 3 * pt_idx[i]
    H_blocks[col_cam:col_cam+6, col_pt:col_pt+3] = H_pl[i]
    H_blocks[col_pt:col_pt+3, col_cam:col_cam+6] = H_pl[i].T # H is symmetric: H_lp = H_pl^T

err_H = np.max(np.abs(H_blocks - H_dense)) / np.max(np.abs(H_dense))
err_g = np.max(np.abs(g_blocks - g_dense)) / np.max(np.abs(g_dense))
print(f"check 4, block H, g vs dense J^T W J: H rel err = {err_H:.1e}, g rel err = {err_g:.1e}  (expect ~1e-15)")
print(f"         dense J is {J_dense.shape}, {100 * np.mean(J_dense != 0):.2f}% non-zero")
