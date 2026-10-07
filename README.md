# P679 – Hourly Energy Consumption Forecast (PJM West)

**Live app:** https://hourly-consumption-forecast-abtva8ingoigzns37dgvyk.streamlit.app/

Forecasts electricity consumption (MW) for the PJM West region using hourly data from 2002-04-01 to 2018-08-03 (143,206 records).

## Approach
1. **Cleaning:** convert types, drop invalid rows, check duplicate and missing timestamps.
2. **EDA:** hourly, weekly and seasonal patterns; outlier analysis (IQR); ADF stationarity test; ACF/PACF.
3. **Modelling:** hourly data aggregated to daily averages. Last 365 days held out as the test set.
   - ARIMA (3,0,3), order chosen by AIC
   - SARIMA, weekly seasonality (s = 7), tuned by AIC
   - XGBoost with calendar, lag (1, 2, 3, 7, 14, 30) and rolling-mean (7, 14, 30) features, tuned with TimeSeriesSplit GridSearchCV
4. **Evaluation:** test period 2017-08-04 to 2018-08-03.

| Model   | MAE    | RMSE   | MAPE (%) | R²     |
|---------|--------|--------|----------|--------|
| ARIMA   | 586.42 | 779.11 | 10.06    | -0.034 |
| SARIMA  | 565.59 | 758.14 | 9.59     | 0.021  |
| **XGBoost** | **455.23** | **602.51** | **7.82** | **0.382** |

XGBoost is the best model on every metric and is the deployed model.

## App
- **Forecast:** 1–90 day recursive forecast, chart, table and CSV download
- **Model Evaluation:** model comparison and live XGBoost test-period results
- **Hourly Pattern:** average load by hour, weekday vs weekend

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `app.py`: Streamlit app
- `PJMW_MW_Hourly.xlsx`: dataset
- `*.ipynb`: EDA, model building and evaluation notebooks
