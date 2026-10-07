import pandas as pd
import streamlit as st
from xgboost import XGBRegressor

st.set_page_config(page_title="PJM Energy Forecast", layout="wide")

FEATURES = [
    "day_of_week", "day_of_month", "month", "quarter", "day_of_year", "week_of_year",
    "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_30",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_30",
]
# Best params from GridSearchCV (TimeSeriesSplit, 3 folds) in the notebook
PARAMS = dict(n_estimators=500, max_depth=3, learning_rate=0.05,
              subsample=0.8, colsample_bytree=0.8)


def make_features(s):
    X = pd.DataFrame(index=s.index)
    X["day_of_week"] = s.index.dayofweek
    X["day_of_month"] = s.index.day
    X["month"] = s.index.month
    X["quarter"] = s.index.quarter
    X["day_of_year"] = s.index.dayofyear
    X["week_of_year"] = s.index.isocalendar().week.astype(int)
    for lag in (1, 2, 3, 7, 14, 30):
        X[f"lag_{lag}"] = s.shift(lag)
    for w in (7, 14, 30):
        X[f"rolling_mean_{w}"] = s.shift(1).rolling(w).mean()
    return X[FEATURES]


@st.cache_data
def load_hourly():
    df = pd.read_excel("PJMW_MW_Hourly.xlsx")
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df["PJMW_MW"] = pd.to_numeric(df["PJMW_MW"], errors="coerce")
    return df.dropna().sort_values("Datetime").set_index("Datetime")["PJMW_MW"]


@st.cache_resource
def train(daily):
    X = make_features(daily)
    mask = X.notna().all(axis=1)
    model = XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1, **PARAMS)
    model.fit(X[mask], daily[mask])
    return model


def forecast(model, daily, days):
    # Recursive: each prediction becomes the lag input for the next day (same as notebook)
    history = daily.copy()
    for _ in range(days):
        nxt = history.index[-1] + pd.Timedelta(days=1)
        ext = pd.concat([history.tail(31), pd.Series([float("nan")], index=[nxt])])
        row = make_features(ext).iloc[[-1]]
        history.loc[nxt] = float(model.predict(row)[0])
    return history.iloc[len(daily):].rename("Forecast_MW")


@st.cache_data
def evaluate(daily):
    # Same 365-day hold-out as the notebook: train before it, forecast it recursively
    train_s, test_s = daily.iloc[:-365], daily.iloc[-365:]
    pred = forecast(train(train_s), train_s, 365)
    pred.index = test_s.index
    err = test_s - pred
    metrics = {
        "MAE": err.abs().mean(),
        "RMSE": (err ** 2).mean() ** 0.5,
        "MAPE (%)": (err.abs() / test_s).mean() * 100,
        "R2": 1 - (err ** 2).sum() / ((test_s - test_s.mean()) ** 2).sum(),
    }
    return pd.concat([test_s.rename("Actual_MW"), pred.rename("Predicted_MW")], axis=1), metrics


# From the notebook's FINAL MODEL COMPARISON (test period 2017-08-04 to 2018-08-03)
COMPARISON = pd.DataFrame(
    {"MAE": [586.42, 565.59, 455.23], "RMSE": [779.11, 758.14, 602.51],
     "MAPE (%)": [10.06, 9.59, 7.82], "R2": [-0.034, 0.021, 0.382]},
    index=pd.Index(["ARIMA (3,0,3)", "SARIMA", "XGBoost"], name="Model"),
)

st.title("P679 – Hourly Energy Consumption Forecast (PJM West)")
st.caption("Final model: XGBoost on daily-average load, selected over ARIMA/SARIMA on a 365-day hold-out test")

hourly = load_hourly()
daily = hourly.resample("D").mean().dropna()
model = train(daily)

days = st.sidebar.slider("Days to forecast", 1, 90, 30)
show_hist = st.sidebar.slider("History to show (days)", 30, 730, 180)

tab_fc, tab_eval, tab_hourly = st.tabs(["Forecast", "Model Evaluation", "Hourly Pattern"])

with tab_fc:
    fc = forecast(model, daily, days)
    c1, c2, c3 = st.columns(3)
    c1.metric("Last actual date", daily.index[-1].strftime("%Y-%m-%d"))
    c2.metric("Avg forecast (MW)", f"{fc.mean():,.0f}")
    c3.metric("Peak forecast (MW)", f"{fc.max():,.0f}", fc.idxmax().strftime("%Y-%m-%d"))
    st.line_chart(pd.concat([daily.tail(show_hist).rename("Actual_MW"), fc], axis=1))
    st.subheader("Forecast table")
    st.dataframe(fc.round(1).to_frame(), width="stretch")
    st.download_button("Download forecast CSV", fc.round(1).to_csv(), "forecast.csv", "text/csv")

with tab_eval:
    st.subheader("Model comparison (from notebook)")
    st.dataframe(COMPARISON.style.highlight_min(subset=["MAE", "RMSE", "MAPE (%)"])
                 .highlight_max(subset=["R2"]).format("{:.2f}"), width="stretch")
    st.caption("XGBoost is best on every metric, so it is the deployed model.")

    st.subheader("XGBoost on the 365-day test period (recomputed live)")
    test_df, m = evaluate(daily)
    cols = st.columns(4)
    for col, (name, val) in zip(cols, m.items()):
        col.metric(name, f"{val:,.3f}" if name == "R2" else f"{val:,.2f}")
    st.line_chart(test_df)

with tab_hourly:
    st.subheader("Average load by hour of day")
    by_hour = hourly.groupby(hourly.index.hour).mean().rename("Avg_MW")
    by_hour.index.name = "Hour"
    st.bar_chart(by_hour)
    st.subheader("Average load by hour: weekday vs weekend")
    kind = pd.Series(hourly.index.dayofweek >= 5, index=hourly.index).map({True: "Weekend", False: "Weekday"})
    st.line_chart(hourly.groupby([hourly.index.hour, kind]).mean().unstack().rename_axis("Hour"))
    st.caption("Load is lowest around 3–5 AM, rises from 7–8 AM and peaks around 7 PM; weekends run lower than weekdays.")
