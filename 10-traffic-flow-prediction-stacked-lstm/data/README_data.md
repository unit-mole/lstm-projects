# Dataset Documentation

## Included sample

`sample_traffic_flow_data.csv` contains **2,160 hourly rows**
covering the final 90 days of the deterministic synthetic traffic dataset
used by the supplied notebook.

## Columns

| Column | Meaning |
|---|---|
| `timestamp` | Hourly observation timestamp |
| `vehicle_count` | Simulated vehicles observed during the hour |
| `avg_speed` | Simulated average traffic speed |
| `occupancy` | Simulated roadway/sensor occupancy between 0 and 1 |
| `weather_severity` | Simulated weather severity between 0 and 1 |
| `congestion_index` | Continuous traffic-congestion target |

Time features such as hour, day of week, weekend, and cyclical sine/cosine
encodings are generated in the preprocessing pipeline.

## Safety and licensing

The sample is synthetic and contains no private road, sensor, vehicle, or
infrastructure identifiers. It is suitable for GitHub and demo use.

A real traffic dataset should only be added when its license permits
redistribution. Remove or anonymize road IDs, sensor IDs, location details,
and operationally sensitive infrastructure information before sharing.

## Compatible upload schema

Uploaded CSV files must contain these six columns:

```text
timestamp
vehicle_count
avg_speed
occupancy
weather_severity
congestion_index
```

The application sorts chronologically, removes duplicate timestamps,
interpolates numeric gaps, and generates the model's time features.
