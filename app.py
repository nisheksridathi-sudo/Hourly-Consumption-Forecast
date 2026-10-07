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
def load_daily():
    df = pd.read_excel("PJMW_MW_Hourly.xlsx")
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df["PJMW_MW"] = pd.to_numeric(df["PJMW_MW"], errors="coerce")
    df = df.dropna().sort_values("Datetime")
    return df.set_index("Datetime")["PJMW_MW"].resample("D").mean().dropna()


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
        ext = pd.concat([history, pd.Series([float("nan")], index=[nxt])])
        row = make_features(ext).iloc[[-1]]
        history.loc[nxt] = float(model.predict(row)[0])
    return history.iloc[len(daily):].rename("Forecast_MW")


st.title("PJM West – Daily Energy Consumption Forecast")
st.caption("Model: XGBoost (selected over ARIMA/SARIMA on a 365-day hold-out test)")

daily = load_daily()
model = train(daily)

days = st.sidebar.slider("Days to forecast", 1, 90, 30)
show_hist = st.sidebar.slider("History to show (days)", 30, 730, 180)

fc = forecast(model, daily, days)

c1, c2, c3 = st.columns(3)
c1.metric("Last actual date", daily.index[-1].strftime("%Y-%m-%d"))
c2.metric("Avg forecast (MW)", f"{fc.mean():,.0f}")
c3.metric("Peak forecast (MW)", f"{fc.max():,.0f}", fc.idxmax().strftime("%Y-%m-%d"))

chart = pd.concat([daily.tail(show_hist).rename("Actual_MW"), fc], axis=1)
st.line_chart(chart)

st.subheader("Forecast table")
st.dataframe(fc.round(1).to_frame(), width="stretch")
st.download_button("Download forecast CSV", fc.round(1).to_csv(), "forecast.csv", "text/csv")
