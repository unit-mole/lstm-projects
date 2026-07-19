# Improvement Roadmap

## Highest-value next steps

1. Replace the synthetic demonstration data with a public, licensed predictive-maintenance dataset such as a turbofan run-to-failure or machine-sensor benchmark, while preserving the same pipeline interfaces.
2. Add operating-condition variables and equipment-family normalization to reduce false positives caused by unit-to-unit baseline shifts.
3. Calibrate thresholds per asset, operating regime, or risk tier instead of relying only on one global threshold.
4. Add model and data drift monitoring, sensor-health checks, and missing-signal alerts.
5. Compare temporal models with PCA, Isolation Forest, dense autoencoders, Temporal Convolutional Networks, and Transformer encoders under identical grouped splits.
6. Evaluate early-warning lead time, event-level recall, alarm persistence, and maintenance cost—not only window-level classification metrics.
7. Add root-cause assistance through sensor contribution, change-point detection, maintenance history, and domain rules without claiming causal diagnosis.
8. Add CI using GitHub Actions for linting, tests, artifact checks, and Streamlit smoke tests.

## Production-readiness gate

A production system would require a documented hazard/risk assessment, sensor validation, maintenance-domain review, threshold governance, audit logging, human escalation, rollback procedures, and monitoring for false negatives and distribution shift.
