from pathlib import Path

data_root_path = Path(r"D:\CV\Data\kitti\dataset")

# Anchored to the repo root so scripts work from any working directory.
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
