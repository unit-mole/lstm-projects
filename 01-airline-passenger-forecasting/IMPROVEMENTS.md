# Improvements Made to the Original Notebook

## What was retained

- The original business objective: monthly airline passenger demand forecasting.
- LSTM-based sequence modeling.
- Future multi-step forecasting.
- Model persistence and reload for inference.
- Historical trend visualization.

## What was corrected

1. **Training-only scaling**  
   The original notebook fitted `MinMaxScaler` on all 144 observations before the split. The rebuilt pipeline fits `StandardScaler` only on training-period seasonal-growth values.

2. **Separate validation and test periods**  
   The original final model used the test set as `validation_data`, allowing early stopping and learning-rate decisions to observe test performance. The rebuilt project uses three chronological periods: 96 months training, 24 months validation, and 24 months untouched testing.

3. **Leakage-safe model selection**  
   Hyperparameter decisions are made with validation data only. Final reported metrics use the untouched test period.

4. **Seasonality-aware target formulation**  
   Instead of asking a small LSTM to learn both an accelerating trend and multiplicative seasonality directly from raw levels, the new model predicts year-over-year log growth. The passenger level is reconstructed from the same month one year earlier.

5. **Smaller architecture for a small dataset**  
   The original stacked 64/32-unit model had 29,857 trainable parameters for only 144 observations. The production model uses a compact 16-unit LSTM plus an 8-unit dense layer to reduce overfitting risk.

6. **Proper baseline comparison**  
   Naive, seasonal-naive, 12-month moving-average, and linear-trend forecasts are evaluated against the same final test period.

7. **Additional evaluation**  
   The project now reports MAE, RMSE, MAPE, and R², plus residual analysis, actual-vs-predicted plots, training curves, and downloadable prediction tables.

8. **Native Keras model format**  
   The original `.h5` model has been replaced by the modern `.keras` format.

9. **Reusable project structure**  
   Notebook logic has been separated into tested modules for preprocessing, feature engineering, sequences, training, evaluation, forecasting, and visualization.

10. **Inference-only Streamlit app**  
    The deployed app loads a pre-trained model and scaler, supports sample/uploaded CSV data, offers 6/12/18/24-month forecasts, and exports results without retraining.
