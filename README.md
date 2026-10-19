# VSLAM

Monocular visual SLAM on KITTI odometry, built from scratch phase by phase (see [VO_SLAM_roadmap.md](VO_SLAM_roadmap.md)).

## Results log

KITTI sequence 00, scored against ground truth with `vslam/metrics.py`.

- **ATE**: translation RMSE after alignment. Sim(3) for monocular runs; SE(3) when scale is undefined.
- **RPE**: per-frame drift (`delta=1`, 0.1 s at 10 Hz), computed on the aligned poses.

Add one row per phase. Don't edit old rows.

| Date       | Phase | Method                | Seq | Align | ATE [m] | RPE trans [m] | RPE rot [deg] | Script           |
|------------|-------|-----------------------|-----|-------|---------|---------------|---------------|------------------|
| 2026-10-18 | P0    | Zero-motion baseline  | 00  | SE(3) | 193.62  | 0.864         | 1.236         | `p0_baseline.py` |

The zero-motion baseline has no Sim(3) score: every estimated position is the same point, so scale is undefined. Unaligned, its ATE is 302.11 m.
