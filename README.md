# VSLAM

Monocular visual SLAM on KITTI odometry, built from scratch phase by phase (see [VO_SLAM_roadmap.md](VO_SLAM_roadmap.md)).

## Results log

KITTI sequence 00, scored against ground truth with `vslam/metrics.py`.

- **ATE**: translation RMSE after alignment. Sim(3) for monocular runs; SE(3) when scale is undefined.
- **RPE**: per-frame drift (`delta=1`, 0.1 s at 10 Hz), computed on the aligned poses. Rotation is given as RMSE and as the median per-frame error, since a few bad frames dominate the RMSE.

Add one row per phase. Don't edit old rows.

| Date       | Phase | Method                | Seq | Align | ATE [m] | RPE trans [m] | RPE rot RMSE [deg] | RPE rot median [deg] | Script           |
|------------|-------|-----------------------|-----|-------|---------|---------------|--------------------|----------------------|------------------|
| 2026-10-18 | P0    | Zero-motion baseline  | 00  | SE(3) | 193.62  | 0.864         | 1.236              | 0.338                | `p0_baseline.py` |
| 2026-10-28 | P2    | Frame-to-frame VO, GT step length, low-motion + rotation checks | 00 | SE(3) | 63.22 | 0.110 | 0.537 | 0.154 | `p2_trajectory.py` |

The zero-motion baseline has no Sim(3) score: every estimated position is the same point, so scale is undefined. Unaligned, its ATE is 302.11 m.
