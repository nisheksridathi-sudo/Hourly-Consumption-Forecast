import altair as alt
import pandas as pd
import streamlit as st
from xgboost import XGBRegressor

st.set_page_config(page_title="PJM Energy Forecast", page_icon="⚡", layout="wide")

FEATURES = [
    "day_of_week", "day_of_month", "month", "quarter", "day_of_year", "week_of_year",
    "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_30",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_30",
]
# Best params from GridSearchCV (TimeSeriesSplit, 3 folds) in the notebook
PARAMS = dict(n_estimators=500, max_depth=3, learning_rate=0.05,
              subsample=0.8, colsample_bytree=0.8)

# Palette: ink = actual data, amber = forecast / peaks (grid warning-light colour)
INK, AMBER, STEEL, RULE = "#14213D", "#E0952A", "#5B6B82", "#DCE2EA"


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


def line_chart(df, colors, x_type="T", height=360):
    # Wide frame -> Altair line chart; colors maps column name -> hex, forecast-type series dashed
    x = df.index.name or "x"
    long = df.reset_index().melt(x, var_name="Series", value_name="MW").dropna()
    names = list(colors)
    dashed = [[5, 4] if n in ("Forecast", "Predicted", "Weekend") else [1, 0] for n in names]
    return (
        alt.Chart(long, height=height).mark_line(strokeWidth=1.8)
        .encode(
            x=alt.X(f"{x}:{x_type}", title=None),
            y=alt.Y("MW:Q", title="MW", scale=alt.Scale(zero=False)),
            color=alt.Color("Series:N", scale=alt.Scale(domain=names, range=list(colors.values())),
                            legend=alt.Legend(orient="top", title=None)),
            strokeDash=alt.StrokeDash("Series:N", scale=alt.Scale(domain=names, range=dashed), legend=None),
            tooltip=[alt.Tooltip(f"{x}:{x_type}"), "Series:N", alt.Tooltip("MW:Q", format=",.0f")],
        )
        .interactive(bind_y=False)
    )


def load_curve_svg(by_hour, w=400, h=150, pad=18, top=26, bottom=40):
    # The signature: the real average 24-hour load curve, drawn in the header
    lo, hi = by_hour.min(), by_hour.max()
    base = h - bottom
    pts = [(pad + i * (w - 2 * pad) / 23, base - (v - lo) / (hi - lo) * (base - top))
           for i, v in enumerate(by_hour)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area = f"{pad},{base} {line} {w - pad},{base}"
    pk, lw = int(by_hour.idxmax()), int(by_hour.idxmin())
    (px, py), (lx, ly) = pts[pk], pts[lw]
    ticks = "".join(
        f'<text x="{pts[i][0]:.1f}" y="{h - 2}" text-anchor="middle">{t}</text>'
        for i, t in ((0, "12 AM"), (6, "6 AM"), (12, "12 PM"), (18, "6 PM"), (23, "11 PM")))
    return (
        f'<svg viewBox="0 0 {w} {h}" class="curve" role="img" '
        f'aria-label="Average load by hour: lowest at {lw}:00, peak at {pk}:00">'
        f'<polygon points="{area}" fill="{AMBER}" fill-opacity="0.12"/>'
        f'<polyline points="{line}" fill="none" stroke="{AMBER}" stroke-width="2.5" stroke-linejoin="round"/>'
        f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.5" fill="{AMBER}"/>'
        f'<text x="{px - 8:.1f}" y="{py - 10:.1f}" text-anchor="end" class="hl">Peak {by_hour.max():,.0f} MW</text>'
        f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="3.5" fill="#fff"/>'
        f'<text x="{lx:.1f}" y="{ly + 17:.1f}" text-anchor="middle" class="hl">Low {by_hour.min():,.0f} MW</text>'
        f'{ticks}</svg>'
    )


st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600&family=Barlow+Condensed:wght@500;600;700&display=swap');
html, body, .stApp, .stMarkdown, p, li, label, input, button {{ font-family: 'Barlow', sans-serif; }}
h1, h2, h3, [data-testid="stMetricValue"], .stTabs button p {{ font-family: 'Barlow Condensed', sans-serif !important; }}
[data-testid="stDecoration"] {{ display: none; }}
.block-container {{ padding-top: 2rem; max-width: 1200px; }}
.hero {{ background: {INK}; color: #fff; border-radius: 6px; padding: 2rem 2.2rem 1.4rem;
  display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.15fr); gap: 2rem; align-items: end; margin-bottom: 1.5rem; }}
.hero h1 {{ color: #fff; font-size: 2.6rem; line-height: 1.05; font-weight: 700; margin: 0 0 .7rem; padding: 0; }}
.hero p {{ color: #C9D3E1; font-size: 1.02rem; line-height: 1.5; margin: 0; max-width: 34em; }}
.hero .curve {{ width: 100%; height: auto; }}
.hero .curve text {{ fill: #8FA0B8; font: 500 12px 'Barlow', sans-serif; }}
.hero .curve text.hl {{ fill: #fff; font: 600 13px 'Barlow', sans-serif; }}
.hero .cap {{ color: #8FA0B8; font-size: .82rem; margin: 0 0 .3rem 18px; }}
@media (max-width: 760px) {{ .hero {{ grid-template-columns: 1fr; padding: 1.4rem; }} .hero h1 {{ font-size: 2rem; }} }}
[data-testid="stMetricValue"] {{ font-size: 2.3rem; font-weight: 600; color: {INK}; }}
[data-testid="stMetricLabel"] p {{ color: {STEEL}; font-size: .9rem; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 1.6rem; border-bottom: 1px solid {RULE}; }}
.stTabs button p {{ font-size: 1.15rem; font-weight: 600; }}
h3 {{ color: {INK}; font-weight: 600; }}
</style>""", unsafe_allow_html=True)

hourly = load_hourly()
daily = hourly.resample("D").mean().dropna()
model = train(daily)
by_hour = hourly.groupby(hourly.index.hour).mean().rename("Avg_MW")
by_hour.index.name = "Hour"

st.markdown(
    '<div class="hero"><div>'
    '<h1>PJM West electricity demand forecast</h1>'
    f'<p>Project P679. Daily load forecast built on {len(hourly):,} hourly readings from '
    f'{hourly.index[0]:%B %Y} to {hourly.index[-1]:%B %Y}. The model is XGBoost, chosen over ARIMA and '
    'SARIMA after it was off by 7.8% on average across a full year it never saw.</p></div>'
    f'<div><p class="cap">Average load across a day</p>{load_curve_svg(by_hour)}</div></div>',
    unsafe_allow_html=True,
)

st.sidebar.header("Forecast settings")
days = st.sidebar.slider("Days to forecast", 1, 90, 30)
show_hist = st.sidebar.slider("History to show (days)", 30, 730, 180)
st.sidebar.caption("The forecast starts the day after the last reading in the dataset "
                   f"({daily.index[-1]:%d %b %Y}).")

tab_fc, tab_eval, tab_hourly = st.tabs(["Forecast", "Model evaluation", "Hourly pattern"])

with tab_fc:
    fc = forecast(model, daily, days)
    c1, c2, c3 = st.columns(3)
    c1.metric("Last actual date", daily.index[-1].strftime("%d %b %Y"))
    c2.metric(f"Average over next {days} days", f"{fc.mean():,.0f} MW")
    c3.metric("Highest forecast day", f"{fc.max():,.0f} MW", fc.idxmax().strftime("%d %b %Y"),
              delta_color="off")
    hist = pd.concat([daily.tail(show_hist).rename("Actual"), fc.rename("Forecast")], axis=1).rename_axis("Date")
    start = alt.Chart(pd.DataFrame({"Date": [fc.index[0]]})).mark_rule(color=STEEL, strokeDash=[2, 3]) \
        .encode(x="Date:T")
    st.altair_chart(line_chart(hist, {"Actual": INK, "Forecast": AMBER}) + start, width="stretch")
    st.subheader("Forecast table")
    table = fc.round(0).astype(int).rename("Forecast (MW)").to_frame()
    table.index = table.index.strftime("%a %d %b %Y")
    st.dataframe(table.rename_axis("Date"), width="stretch")
    st.download_button("Download forecast CSV", fc.round(1).to_csv(), "forecast.csv", "text/csv")

with tab_eval:
    st.subheader("Model comparison on the test year")
    st.caption("All three models forecast 4 Aug 2017 to 3 Aug 2018 without seeing it. Best value in each column is highlighted.")
    best = "background-color: #FBE7C6; font-weight: 600"
    st.dataframe(COMPARISON.style.highlight_min(subset=["MAE", "RMSE", "MAPE (%)"], props=best)
                 .highlight_max(subset=["R2"], props=best).format("{:.2f}"), width="stretch")

    st.subheader("XGBoost on the test year, recomputed live")
    test_df, m = evaluate(daily)
    cols = st.columns(4)
    for col, (name, val) in zip(cols, m.items()):
        col.metric(name, f"{val:,.3f}" if name == "R2" else f"{val:,.2f}")
    st.altair_chart(line_chart(test_df.set_axis(["Actual", "Predicted"], axis=1).rename_axis("Date"), {"Actual": INK, "Predicted": AMBER}),
                    width="stretch")

with tab_hourly:
    st.subheader("Average load by hour of day")
    bars = alt.Chart(by_hour.reset_index(), height=320).mark_bar().encode(
        x=alt.X("Hour:O", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Avg_MW:Q", title="MW", scale=alt.Scale(zero=False)),
        color=alt.condition(alt.datum.Avg_MW == by_hour.max(), alt.value(AMBER), alt.value(INK)),
        tooltip=["Hour:O", alt.Tooltip("Avg_MW:Q", format=",.0f")],
    )
    st.altair_chart(bars, width="stretch")
    st.subheader("Weekday vs weekend")
    kind = pd.Series(hourly.index.dayofweek >= 5, index=hourly.index).map({True: "Weekend", False: "Weekday"})
    wk = hourly.groupby([hourly.index.hour, kind]).mean().unstack().rename_axis("Hour")
    st.altair_chart(line_chart(wk, {"Weekday": INK, "Weekend": AMBER}, x_type="O", height=320), width="stretch")
    st.caption("Load is lowest around 3–5 AM, rises from 7–8 AM and peaks around 7 PM; weekends run lower than weekdays.")
