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
| 2026-10-29 | P3    | Mono VO, PnP + local map, GT scale at init only (**failed checkpoint**) | 00 | Sim(3) | 189.81 | 0.845 | 1.213 | 0.415 | `p3_PnP.py` |
| 2026-11-01 | P4    | Stereo VO (SGBM depth + frame-to-frame PnP), no GT used | 00 | SE(3) | 29.36 | 0.031 | 0.120 | 0.050 | `p4_stereo.py` |

The zero-motion baseline has no Sim(3) score: every estimated position is the same point, so scale is undefined. Unaligned, its ATE is 302.11 m.

**P3 notes.** Scale held near 1 for about 150 frames, then collapsed in steps (2–8× each) to about 1e-5 by frame 1100. After that, re-initializations inherited the tiny scale, and 1904 frames were lost (946 re-inits). Tracking a correct map works: PnP against a map built from GT poses gives steps within 0.94–1.08 of GT. The drift comes from each keyframe handoff passing baseline errors into new two-view landmarks, with nothing to pull the scale back. Local BA (P5) or metric depth (P4) should fix it.

**P4 notes.** No lost frames. Scale stays metric: the estimated/GT step ratio has a median of 0.99–1.00 in every quarter of the run. Sim(3) alignment barely helps (28.98 m vs 29.36 m SE(3)), so the remaining error is not scale. It is slow heading drift: the trajectory has the right shape but is gradually rotated. Unaligned ATE is 67.29 m. Side-by-side plot: `results/p4_mono_vs_stereo_00.png`.
