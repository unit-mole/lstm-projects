# ECG Demonstration Data

The included `sample_ecg_signals.csv` contains **240 synthetic, privacy-safe ECG-like sequences**.
It does not contain real patient data, protected health information, medical record identifiers,
or clinical diagnoses.

## File structure

Each row is one fixed-length signal:

| Field | Description |
|---|---|
| `signal_id` | Synthetic record identifier |
| `label` | `0` for normal and `1` for synthetic anomaly |
| `anomaly_type` | Synthetic generation mode |
| `sample_000`–`sample_139` | 140 signal amplitudes |

## Synthetic anomaly types

- `high_noise`
- `attenuated_amplitude`
- `temporal_shift`
- `localized_spike`

These generated patterns are intentionally simple and separable. They are useful for demonstrating
anomaly-detection engineering but do not represent the full complexity of clinical ECG morphology.

## Compatible uploads

The Streamlit app expects one row per signal and 140 numeric signal columns. Signal columns may use
prefixes such as:

- `sample_`
- `signal_`
- `value_`
- `timestep_`

When no recognized prefix is present, the app uses the first 140 numeric non-metadata columns.

Optional metadata aliases:

- labels: `label`, `target`, `is_anomaly`, `anomaly`, `class`
- record IDs: `signal_id`, `record_id`, `id`, `sample_id`
- types: `anomaly_type`, `signal_type`, `type`

## Data safety

Never upload private patient data, protected health information, confidential medical records, or
data that you are not authorized to process. Real ECG data requires governance, de-identification,
patient-level splitting, clinical review, and applicable regulatory controls.
