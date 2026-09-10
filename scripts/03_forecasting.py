from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "accident_data_v1.0.0_2023.db"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

if not DB_PATH.exists():
    raise FileNotFoundError(f"Database not found: {DB_PATH}")

with sqlite3.connect(DB_PATH) as con:
    accidents = pd.read_sql_query("""
        SELECT accident_index, accident_year, police_force, date, lsoa_of_accident_location
        FROM accident WHERE accident_year IN (2017, 2018, 2019)
    """, con)

accidents["date"] = pd.to_datetime(accidents["date"], dayfirst=True, errors="coerce")
accidents = accidents.dropna(subset=["date"])

# ============================================================
# WEEKLY ARIMA FORECASTS BY POLICE FORCE
# ============================================================

train = accidents[accidents["accident_year"].isin([2017, 2018])]
test = accidents[accidents["accident_year"] == 2019]
top_forces = train["police_force"].value_counts().head(3).index.tolist()
force_names = {1:"Metropolitan Police", 20:"West Midlands", 46:"Kent"}

results = []

for code in top_forces:
    name = force_names.get(code, f"Police Force {code}")
    train_series = train[train["police_force"] == code].set_index("date").resample("W-SUN").size().asfreq("W-SUN", fill_value=0)
    test_series = test[test["police_force"] == code].set_index("date").resample("W-SUN").size().asfreq("W-SUN", fill_value=0)

    adf_p = adfuller(train_series)[1]
    d = 1 if adf_p > 0.05 else 0
    orders = [(0,d,0), (1,d,0), (0,d,1), (1,d,1), (2,d,1), (1,d,2)]

    best_model, best_order, best_aic = None, None, np.inf
    for order in orders:
        try:
            fitted = ARIMA(train_series, order=order).fit()
            if fitted.aic < best_aic:
                best_model, best_order, best_aic = fitted, order, fitted.aic
        except Exception:
            continue

    forecast_obj = best_model.get_forecast(steps=len(test_series))
    forecast = pd.Series(forecast_obj.predicted_mean.values, index=test_series.index).clip(lower=0)
    ci = forecast_obj.conf_int()
    ci.index = test_series.index

    mae = mean_absolute_error(test_series, forecast)
    rmse = np.sqrt(mean_squared_error(test_series, forecast))
    actual_total, predicted_total = test_series.sum(), forecast.sum()
    percentage_error = ((predicted_total - actual_total) / actual_total) * 100
    ljung_p = acorr_ljungbox(best_model.resid, lags=[10], return_df=True)["lb_pvalue"].iloc[0]

    results.append({
        "police_force": name, "best_order": str(best_order), "ADF_pvalue": round(adf_p,4),
        "MAE": round(mae,2), "RMSE": round(rmse,2), "actual_total": int(actual_total),
        "predicted_total": int(round(predicted_total)), "percentage_error": round(percentage_error,2),
        "ljung_box_pvalue": round(ljung_p,4)
    })

    plt.figure(figsize=(12,5))
    plt.plot(test_series.index, test_series, label="Actual 2019")
    plt.plot(forecast.index, forecast, "--", label="Predicted 2019")
    plt.fill_between(test_series.index, ci.iloc[:,0].clip(lower=0), ci.iloc[:,1], alpha=0.2)
    plt.title(f"ARIMA Weekly Accident Forecast - {name}")
    plt.xlabel("Week")
    plt.ylabel("Accidents")
    plt.legend()
    plt.tight_layout()
    filename = name.lower().replace(" ", "_") + "_forecast.png"
    plt.savefig(RESULTS_DIR / filename, dpi=300)
    plt.close()

pd.DataFrame(results).to_csv(RESULTS_DIR / "police_force_forecast_metrics.csv", index=False)


# WEST YORKSHIRE DAILY SARIMAX FORECAST


wy = accidents[(accidents["accident_year"] == 2018) & (accidents["police_force"] == 13)].copy()
wy = wy[wy["lsoa_of_accident_location"].notna() & (wy["lsoa_of_accident_location"] != "-1")]

top30 = wy[(wy["date"] >= "2018-01-01") & (wy["date"] <= "2018-03-31")]["lsoa_of_accident_location"].value_counts().head(30).index
wy = wy[wy["lsoa_of_accident_location"].isin(top30)]

daily = wy.set_index("date").resample("D").size().reindex(pd.date_range("2018-01-01", "2018-07-31"), fill_value=0)
train_series = daily["2018-01-01":"2018-06-30"]
test_series = daily["2018-07-01":"2018-07-31"]

def make_exog(index):
    return pd.DataFrame({"day_of_week":index.dayofweek, "is_weekend":index.dayofweek.isin([5,6]).astype(int)}, index=index)

adf_p = adfuller(train_series)[1]
d = 1 if adf_p > 0.05 else 0
train_exog, test_exog = make_exog(train_series.index), make_exog(test_series.index)

orders = [(0,d,1), (1,d,0), (1,d,1)]
seasonal_orders = [(0,0,0,0), (1,0,0,7), (0,0,1,7), (1,0,1,7)]
best_model, best_order, best_seasonal, best_aic = None, None, None, np.inf

for order in orders:
    for seasonal in seasonal_orders:
        try:
            fitted = SARIMAX(train_series, exog=train_exog, order=order, seasonal_order=seasonal,
                             enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            if fitted.aic < best_aic:
                best_model, best_order, best_seasonal, best_aic = fitted, order, seasonal, fitted.aic
        except Exception:
            continue

forecast_obj = best_model.get_forecast(steps=len(test_series), exog=test_exog)
forecast = pd.Series(forecast_obj.predicted_mean.values, index=test_series.index).clip(lower=0)
ci = forecast_obj.conf_int()
ci.index = test_series.index

plt.figure(figsize=(12,5))
plt.plot(test_series.index, test_series, label="Actual July 2018")
plt.plot(forecast.index, forecast, "--", label="Predicted July 2018")
plt.fill_between(test_series.index, ci.iloc[:,0].clip(lower=0), ci.iloc[:,1], alpha=0.2)
plt.title("West Yorkshire SARIMAX Daily Accident Forecast")
plt.xlabel("Date")
plt.ylabel("Daily Accidents")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "west_yorkshire_sarimax_forecast.png", dpi=300)
plt.close()

print(pd.DataFrame(results))
print("\nWest Yorkshire SARIMAX")
print("Order:", best_order, "Seasonal:", best_seasonal)
print("Actual July total:", int(test_series.sum()))
print("Predicted July total:", int(round(forecast.sum())))
