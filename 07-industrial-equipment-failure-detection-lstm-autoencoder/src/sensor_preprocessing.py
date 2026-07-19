from __future__ import annotations

import pandas as pd

from .data_preprocessing import DatasetSchema, apply_scaler, clean_sensor_data


def prepare_inference_frame(
    frame: pd.DataFrame,
    scaler,
    metadata: dict,
    selected_unit=None,
) -> tuple[pd.DataFrame, pd.DataFrame, DatasetSchema]:
    """Apply the exact training-time schema, cleaning, filtering, and scaling."""
    schema = DatasetSchema(
        sensor_cols=list(metadata["sensor_cols"]),
        unit_id_col=metadata.get("unit_id_col", "unit_id"),
        time_col=metadata.get("time_col", "cycle"),
        label_col=metadata.get("label_col"),
    )
    clean = clean_sensor_data(frame, schema)
    if selected_unit is not None:
        clean = clean[clean[schema.unit_id_col] == selected_unit].copy()
        if clean.empty:
            raise ValueError(f"No rows found for equipment ID {selected_unit!r}.")
    scaled = apply_scaler(clean, schema.sensor_cols, scaler)
    return clean.reset_index(drop=True), scaled.reset_index(drop=True), schema
