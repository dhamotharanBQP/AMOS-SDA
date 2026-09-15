import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


POSITION_COLUMNS = ["x_eme2000_km", "y_eme2000_km", "z_eme2000_km"]
VELOCITY_COLUMNS = ["vx_eme2000_km_s", "vy_eme2000_km_s", "vz_eme2000_km_s"]
REQUIRED_COLUMNS = ["timestamp", "satellite", *POSITION_COLUMNS, *VELOCITY_COLUMNS]


class EphemerisDataset(Dataset):
    def __init__(self, csv_file_path, n_samples_per_epoch, window_duration_sec=500.0):
        self.csv_file_path = str(csv_file_path)
        self.n_samples = int(n_samples_per_epoch)
        self.window_duration_sec = float(window_duration_sec)
        if self.n_samples <= 0:
            raise ValueError("n_samples_per_epoch must be positive")
        if self.window_duration_sec <= 0:
            raise ValueError("window_duration_sec must be positive")

        df = pd.read_csv(self.csv_file_path)
        missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
        if missing:
            raise ValueError(f"Missing required CSV columns: {', '.join(missing)}")

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
        start = df["timestamp"].min()
        df["time_sec"] = (df["timestamp"] - start).dt.total_seconds()
        df[POSITION_COLUMNS] = df[POSITION_COLUMNS].astype(np.float64) * 1000.0
        df[VELOCITY_COLUMNS] = df[VELOCITY_COLUMNS].astype(np.float64) * 1000.0

        self.satellite_data = []
        for satellite, sat_df in df.groupby("satellite", sort=False):
            sat_df = sat_df.sort_values("time_sec")
            times = sat_df["time_sec"].to_numpy(dtype=np.float64)
            if len(times) < 2 or times[-1] - times[0] < self.window_duration_sec:
                continue
            candidates = np.flatnonzero(times <= times[-1] - self.window_duration_sec)
            valid_starts = candidates[
                times[candidates + 1] <= times[candidates] + self.window_duration_sec
            ]
            if not len(valid_starts):
                continue
            self.satellite_data.append(
                {
                    "name": satellite,
                    "time_sec": times,
                    "pos_m": sat_df[POSITION_COLUMNS].to_numpy(dtype=np.float64),
                    "vel_m_s": sat_df[VELOCITY_COLUMNS].to_numpy(dtype=np.float64),
                    "valid_starts": valid_starts,
                }
            )

        if not self.satellite_data:
            raise ValueError(
                f"No satellite contains a {self.window_duration_sec:g}-second window"
            )
        print(
            f"[Data] Loaded {len(self.satellite_data)} usable satellites from "
            f"{self.csv_file_path}"
        )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, _):
        sat = self.satellite_data[np.random.randint(len(self.satellite_data))]
        start_index = int(np.random.choice(sat["valid_starts"]))
        start_time = sat["time_sec"][start_index]
        end_time = start_time + self.window_duration_sec
        indices = np.flatnonzero(
            (sat["time_sec"] > start_time) & (sat["time_sec"] <= end_time)
        )
        initial_state = np.concatenate(
            (sat["pos_m"][start_index], sat["vel_m_s"][start_index])
        )
        return (
            initial_state,
            self.window_duration_sec,
            sat["time_sec"][indices] - start_time,
            sat["pos_m"][indices],
            sat["vel_m_s"][indices],
        )


def _empty(rows, columns, device, dtype):
    return torch.empty(rows, columns, device=device, dtype=dtype)


def ephemeris_collate_fn(batch, K_pde, device, dtype):
    data_parts = []
    pde_parts = []
    k_pde = int(K_pde)

    for initial_state, duration, data_times, positions, velocities in batch:
        r0 = torch.as_tensor(initial_state[:3], device=device, dtype=dtype).view(1, 3)
        v0 = torch.as_tensor(initial_state[3:], device=device, dtype=dtype).view(1, 3)
        count = len(data_times)
        if count:
            data_parts.append(
                (
                    torch.as_tensor(data_times, device=device, dtype=dtype).view(-1, 1),
                    r0.expand(count, -1),
                    v0.expand(count, -1),
                    torch.as_tensor(positions, device=device, dtype=dtype),
                    torch.as_tensor(velocities, device=device, dtype=dtype),
                    torch.full((count, 1), duration, device=device, dtype=dtype),
                )
            )
        if k_pde > 0:
            times = torch.rand(k_pde, 1, device=device, dtype=dtype)
            times = times.mul(duration).clamp(min=1e-6 * duration, max=duration)
            pde_parts.append(
                (
                    times,
                    r0.expand(k_pde, -1),
                    v0.expand(k_pde, -1),
                    torch.full((k_pde, 1), duration, device=device, dtype=dtype),
                )
            )

    if data_parts:
        data = tuple(torch.cat(items, dim=0) for items in zip(*data_parts))
    else:
        data = (
            _empty(0, 1, device, dtype),
            _empty(0, 3, device, dtype),
            _empty(0, 3, device, dtype),
            _empty(0, 3, device, dtype),
            _empty(0, 3, device, dtype),
            _empty(0, 1, device, dtype),
        )

    if pde_parts:
        pde = tuple(torch.cat(items, dim=0) for items in zip(*pde_parts))
    else:
        pde = (
            _empty(0, 1, device, dtype),
            _empty(0, 3, device, dtype),
            _empty(0, 3, device, dtype),
            _empty(0, 1, device, dtype),
        )

    t_data, x0_data, u0_data, r_truth, v_truth, tseg_data = data
    t_pde, x0_pde, u0_pde, tseg_pde = pde
    return (
        t_data,
        x0_data,
        u0_data,
        r_truth,
        v_truth,
        t_pde,
        x0_pde,
        u0_pde,
        tseg_data,
        tseg_pde,
    )
