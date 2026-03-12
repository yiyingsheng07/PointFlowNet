import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial import cKDTree


class PointDataset(Dataset):
    def __init__(self,
                 filepath: str,
                 filenames: list[str],
                 model_choice: int,
                 pv_choice: int,
                 random_points: bool = True,
                 norm_stats_path: str = None):
        self.filepath = filepath
        self.filenames = filenames
        self.model_choice = int(model_choice)
        self.pv_choice = int(pv_choice)
        self.random_points = bool(random_points)
        self._wall_tree_cache: dict[str, cKDTree] = {}
        self.norm_stats_path = norm_stats_path
        self.mean_xyz = None
        self.std_scalar = None

        if self.norm_stats_path is not None:
            stats = np.load(self.norm_stats_path)
            self.mean_xyz = stats["mean_xyz"].astype(np.float32)      # (3,)
            self.std_scalar = float(stats["std_scalar"])
            if (not np.isfinite(self.std_scalar)) or self.std_scalar < 1e-12:
                self.std_scalar = 1.0

    # -------------------------------------------------------------
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: int):
        fname = self.filenames[idx]
        wall_csv = os.path.join(self.filepath, f"{fname}_wall.csv")
        pv_csv   = os.path.join(self.filepath, f"{fname}_fluiddomain.csv")

        if not os.path.exists(wall_csv):
            raise FileNotFoundError(f"Missing wall file: {wall_csv}")
        if not os.path.exists(pv_csv):
            raise FileNotFoundError(f"Missing fluid-domain file: {pv_csv}")

        # ---------- load wall cloud & build (or get) KD‑tree ----------
        wall_xyz_full = np.loadtxt(wall_csv, delimiter=",", skiprows=1, usecols=(0, 1, 2), dtype=np.float32)
        wall_xyz_full = wall_xyz_full.reshape(-1, 3)  # guard for single‑row case
        if fname not in self._wall_tree_cache:
            self._wall_tree_cache[fname] = cKDTree(wall_xyz_full)
        wall_tree = self._wall_tree_cache[fname]

        # ---------- sample surface points ----------
        surf_idx = self._choose_indices(len(wall_xyz_full), self.model_choice)
        m_xyz = wall_xyz_full[surf_idx]                                   # (N₁,3)

        # ---------- load full interior cloud & sample ----------
        pv_full = np.loadtxt(pv_csv, delimiter=",", skiprows=1, dtype=np.float32)
        pv_full = pv_full.reshape(-1, pv_full.shape[-1])
        xyz_full = pv_full[:, 0:3]
        vel_full = pv_full[:, -3:]
        pv_idx = self._choose_indices(len(xyz_full), self.pv_choice)
        pv_xyz = xyz_full[pv_idx]                                         # (N₂,3)
        pv_vel = vel_full[pv_idx]                                         # (N₂,3)

        # ---------- wall distance for sampled interior points ----------
        dist, _ = wall_tree.query(pv_xyz, k=1)
        wall_dist = dist.astype(np.float32)                               # (N₂,)


        # ---------- normalize xyz & distance ----------
        if self.mean_xyz is None:
            # fallback (if norm_stats_path not provided): keep your old behavior
            scale = np.max(np.abs(pv_xyz))
            if scale == 0: scale = 1.0
            m_xyz_norm  = m_xyz / scale
            pv_xyz_norm = pv_xyz / scale
            dist_norm   = wall_dist / scale
        else:
            m_xyz_norm  = (m_xyz - self.mean_xyz[None, :]) / self.std_scalar
            pv_xyz_norm = (pv_xyz - self.mean_xyz[None, :]) / self.std_scalar
            dist_norm   = wall_dist / self.std_scalar

        # ---------- assemble tensors (avoid from_numpy) ----------
        model_tensor   = torch.tensor(m_xyz_norm.copy(), dtype=torch.float32)                 # (N₁,3)
        pv_feat_tensor = torch.tensor(np.hstack([pv_xyz_norm, dist_norm[:, None]]), dtype=torch.float32)  # (N₂,4)
        pv_label_tensor = torch.tensor(pv_vel.copy(), dtype=torch.float32)                   # (N₂,3)
        ori_pv_xyz_tensor = torch.tensor(pv_xyz.copy(), dtype=torch.float32)                 # (N₂,3)

        return fname, model_tensor, pv_feat_tensor, pv_label_tensor, ori_pv_xyz_tensor

    # -------------------------------------------------------------
    def _choose_indices(self, n_total: int, n_target: int):
        if n_target >= n_total:
            return np.arange(n_total, dtype=np.int64)
        if self.random_points:
            return np.array(random.sample(range(n_total), n_target), dtype=np.int64)
        return np.array(random.Random(42).sample(range(n_total), n_target), dtype=np.int64)

