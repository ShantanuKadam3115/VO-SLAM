import numpy as np
from config import data_root_path


calib = {}
for line in open(data_root_path / "sequences" / "00" / "calib.txt", "r", encoding="utf-8"):
    key, vals = line.split(":", 1)
    calib[key] = np.array(vals.split(), dtype=float).reshape(3, 4)

P0 = calib["P0"]
K  = P0[:, :3]
B  = -calib["P1"][0, 3] / calib["P1"][0, 0]   # 0.537 m

print(K)