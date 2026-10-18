from pathlib import Path

import matplotlib.pyplot as plt


def plot_trajectories(trajs, labels, title, save_path=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    for T, label in zip(trajs, labels, strict=True):
        x = T[:, 0, 3]
        z = T[:, 2, 3]
        (line,) = ax.plot(x, z, label=label, linewidth=1.2)
        ax.scatter(x[0], z[0], color=line.get_color(), marker="o", s=40, zorder=3)
        ax.scatter(x[-1], z[-1], color=line.get_color(), marker="x", s=40, zorder=3)

    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax
