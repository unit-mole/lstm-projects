from __future__ import annotations

import numpy as np


def generate_ecg_dataset(
    n_normal: int = 2200,
    n_anomaly: int = 800,
    sequence_length: int = 140,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate the exact synthetic ECG-like signals used by the supplied notebook."""
    rng = np.random.default_rng(seed)
    time_axis = np.linspace(0, 1, sequence_length)

    def normal_wave() -> np.ndarray:
        base = 0.15 * np.sin(2 * np.pi * 5 * time_axis)
        qrs = np.exp(-((time_axis - 0.45) ** 2) / 0.0009) * 1.4
        p_wave = np.exp(-((time_axis - 0.2) ** 2) / 0.003) * 0.18
        t_wave = np.exp(-((time_axis - 0.7) ** 2) / 0.01) * 0.28
        noise = rng.normal(0, 0.03, sequence_length)
        return (base + qrs + p_wave + t_wave + noise).astype("float32")

    anomaly_types: list[str] = []

    def anomaly_wave() -> np.ndarray:
        wave = normal_wave()
        mode = int(rng.integers(0, 4))
        if mode == 0:
            wave += rng.normal(0, 0.16, sequence_length)
            anomaly_type = "high_noise"
        elif mode == 1:
            wave *= rng.uniform(0.4, 0.7)
            anomaly_type = "attenuated_amplitude"
        elif mode == 2:
            wave = np.roll(wave, int(rng.integers(8, 20)))
            anomaly_type = "temporal_shift"
        else:
            spike_position = int(rng.integers(10, sequence_length - 10))
            wave[spike_position : spike_position + 3] += rng.uniform(0.8, 1.5)
            anomaly_type = "localized_spike"
        anomaly_types.append(anomaly_type)
        return wave.astype("float32")

    normal = np.asarray([normal_wave() for _ in range(n_normal)], dtype="float32")
    anomaly = np.asarray([anomaly_wave() for _ in range(n_anomaly)], dtype="float32")

    signals = np.concatenate([normal, anomaly], axis=0)[..., None]
    labels = np.concatenate(
        [np.zeros(len(normal)), np.ones(len(anomaly))]
    ).astype(int)
    types = np.asarray(["normal"] * len(normal) + anomaly_types, dtype=object)
    return signals, labels, types
