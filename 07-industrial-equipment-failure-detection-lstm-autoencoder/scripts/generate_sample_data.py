from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.synthetic_data import generate_predictive_maintenance_data

frame, _ = generate_predictive_maintenance_data()
sample = frame[frame["unit_id"].isin([104, 105, 106, 108, 110, 120])]
output = PROJECT_ROOT / "data" / "sample_equipment_sensor_data.csv"
sample.to_csv(output, index=False)
print(f"Saved {len(sample):,} rows to {output}")
