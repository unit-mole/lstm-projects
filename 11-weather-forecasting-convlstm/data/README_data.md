# Data Documentation

The bundled dataset is a deterministic, synthetic spatiotemporal weather-map dataset created for portfolio demonstration. No restricted, private, or operational meteorological data is included.

## Files

- `sample_weather_sequences.npz`: 24 samples with `X`, next-frame `y`, and six-step `future_y`.
- `sample_weather_sequence.npy`: one input sequence with shape `6 × 24 × 24 × 1`.
- `sample_weather_target.npy`: the next frame for the single sample.
- `sample_weather_grid.csv`: long-form representation of the six input maps for reviewers who prefer tabular inspection.

Values are normalized to `[0, 1]` and represent synthetic weather or precipitation intensity—not physical units. Replace this data only with a dataset whose license and redistribution terms permit GitHub publication. For real weather data, split chronologically and fit normalization using training periods only.
