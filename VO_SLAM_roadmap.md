# Roadmap: Visual Odometry to Mini-SLAM from Scratch

**Goal:** Build a monocular/stereo visual odometry (VO) system, then grow it into a small feature-based SLAM system (keyframes, local bundle adjustment, loop closure). You write all the code; I explain, review and debug.

**Dataset:** KITTI Odometry (grayscale sequences 00-10 have ground truth). Start with one short sequence. Later optional: TUM RGB-D or EuRoC.

**Allowed libraries:** NumPy, OpenCV (feature detection/matching, `findEssentialMat`, `recoverPose`, `solvePnPRansac` only after you understand them), matplotlib, SciPy (`least_squares` for your first bundle adjustment). Avoid ORB-SLAM/OpenVSLAM-style libraries; the point is to build it.

---

## Phase 0: Setup and Evaluation Tooling
*Build the yardstick before the system.*

- [ ] Download one KITTI odometry sequence (images, calibration, ground-truth poses)
- [ ] Parse calibration (`P0`/`P1`) and ground-truth poses (3x4 matrices per frame)
- [ ] Trajectory plotter: top-down (x-z) view of estimated vs ground truth
- [ ] Metrics: Absolute Trajectory Error (ATE) and Relative Pose Error (RPE), with a trajectory alignment step (Umeyama/Sim3 for monocular)

**Checkpoint:** Plot the ground-truth trajectory and compute ATE of "robot never moves" as your baseline number.

---

## Phase 1: Two-View Geometry
*The mathematical core of VO.*

- [ ] Feature detection and matching between two frames (ORB or SIFT; ratio test, cross-check)
- [ ] Epipolar geometry: fundamental vs essential matrix, what the epipolar constraint means
- [ ] Estimate E with RANSAC, decompose into R, t, and resolve the 4-way ambiguity with the cheirality check
- [ ] Understand why monocular translation is only known up to scale
- [ ] Triangulate the matched points (DLT) and check reprojection error

**Checkpoint:** For two consecutive KITTI frames, your R and t direction agree with ground truth (rotation error under about 1 degree), and triangulated points lie in front of both cameras.

---

## Phase 2: Frame-to-Frame Monocular VO
*Chain the relative poses into a trajectory.*

- [ ] Loop over the sequence: match, estimate E, recover relative pose, accumulate
- [ ] Fix scale using ground-truth speed for now (acknowledge this is cheating, to be removed later)
- [ ] Handle failure cases: too few matches, near-zero motion, pure rotation (E degenerates)
- [ ] Plot the trajectory against ground truth and watch drift accumulate

**Checkpoint:** A recognizable trajectory on the first 200-500 frames, with ATE reported.

---

## Phase 3: Better VO with PnP and a Local Map
*Use the 2D-3D machinery you already know from ACE.*

- [ ] Triangulate landmarks from the first pair to initialize a small 3D map
- [ ] Track features into new frames (KLT optical flow or descriptor matching)
- [ ] Estimate each new pose with PnP + RANSAC against the map (your ACE inference step, with classical correspondences)
- [ ] Triangulate new landmarks as the camera moves; cull landmarks that fail reprojection checks
- [ ] This also fixes scale consistency, since scale now propagates through the map

**Checkpoint:** Noticeably less scale drift than Phase 2 over a full sequence, with no ground-truth scale injected after initialization.

---

## Phase 4: Stereo Variant (optional but valuable)
*Metric scale for free.*

- [ ] Compute disparity/depth from the KITTI stereo pair (block matching first)
- [ ] Use stereo-triangulated landmarks for PnP so scale is metric from frame 1
- [ ] Compare ATE against your monocular system

**Checkpoint:** Side-by-side trajectory plots: mono vs stereo vs ground truth.

---

## Phase 5: Keyframes and Local Bundle Adjustment
*From odometry to SLAM.*

- [ ] Keyframe selection policy (parallax, tracked-feature count, or distance thresholds)
- [ ] Covisibility: which keyframes observe which landmarks
- [ ] Understand bundle adjustment: the reprojection-error objective, parameters (poses + points), sparsity
- [ ] Implement a small BA on a sliding window using `scipy.optimize.least_squares` with a sparse Jacobian pattern
- [ ] Add robust loss (Huber) so outlier matches do not dominate

**Checkpoint:** BA reduces reprojection error and visibly reduces drift versus Phase 3 on the same sequence.

---

## Phase 6: Loop Closure
*Recognize places and correct drift globally.*

- [ ] Place recognition: bag-of-visual-words (DBoW-style) or a simple global-descriptor retrieval
- [ ] Geometric verification of loop candidates (match, E/PnP, inlier count)
- [ ] Pose graph: nodes = keyframe poses, edges = relative-pose constraints (odometry + loop closure)
- [ ] Pose graph optimization (start with 2D/SE(2) on a toy example, then SE(3) or Sim(3))
- [ ] Apply the correction to keyframes and landmarks

**Checkpoint:** On a KITTI sequence with a loop (e.g. 00 or 05), the trajectory closes the loop after optimization and ATE drops clearly.

---

## Phase 7: Relocalization and Integration
*Tie it back to ACE.*

- [ ] Add tracking-loss recovery: relocalize against the existing map using descriptors + PnP
- [ ] (Stretch) Use your ACE-style scene coordinate network as an alternative relocalizer and compare with the classical one
- [ ] Clean up the code into modules: `frontend/`, `map/`, `backend/`, `eval/`

**Checkpoint:** Deliberately drop a chunk of frames, and the system recovers and continues.

---

## Phase 8: Evaluation and Write-up
- [ ] Run on several KITTI sequences; report ATE/RPE for each stage (VO, +BA, +loop closure)
- [ ] Ablations: feature type, keyframe policy, BA window size, robust loss
- [ ] Compare qualitatively with ORB-SLAM results published for the same sequences
- [ ] Short README with trajectory plots; this becomes a portfolio piece

---

## Extensions (pick later)
- Add IMU pre-integration and move toward visual-inertial odometry (EKF first)
- Dense mapping (depth fusion) or an occupancy map for navigation
- Run on a real robot or ROS 2 bag recordings
- Learned components: SuperPoint/SuperGlue features, learned depth for monocular scale

---

## How We'll Work
1. You write the code for each phase and share it; I review and explain.
2. I point at concepts, function signatures and bugs rather than giving full solutions.
3. We move on only when the checkpoint genuinely holds, with numbers or plots, not just "it runs."
4. Build the evaluation tooling first (Phase 0) so every later improvement is measured, not guessed.
