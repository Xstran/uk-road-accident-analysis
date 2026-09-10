from pathlib import Path
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "accident_data_v1.0.0_2023.db"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

if not DB_PATH.exists():
    raise FileNotFoundError(f"Database not found: {DB_PATH}")

with sqlite3.connect(DB_PATH) as con:
    accident = pd.read_sql_query("SELECT * FROM accident WHERE accident_year = 2018", con)
    vehicle = pd.read_sql_query("SELECT * FROM vehicle WHERE accident_year = 2018", con)
    casualty = pd.read_sql_query("SELECT * FROM casualty WHERE accident_year = 2018", con)

accident = accident.copy()
accident["date"] = pd.to_datetime(accident["date"], dayfirst=True, errors="coerce")
accident["hour"] = pd.to_datetime(accident["time"], format="%H:%M", errors="coerce").dt.hour

day_map = {1:"Sunday", 2:"Monday", 3:"Tuesday", 4:"Wednesday", 5:"Thursday", 6:"Friday", 7:"Saturday"}
day_order = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]
accident["day_name"] = accident["day_of_week"].map(day_map)

# Overall accident patterns
hourly = accident["hour"].value_counts().sort_index()
plt.figure(figsize=(12,5))
plt.bar(hourly.index, hourly.values)
plt.title("Road Accidents by Hour of Day - 2018")
plt.xlabel("Hour")
plt.ylabel("Accidents")
plt.xticks(range(24))
plt.tight_layout()
plt.savefig(RESULTS_DIR / "accidents_by_hour.png", dpi=300)
plt.close()

daily = accident["day_name"].value_counts().reindex(day_order)
plt.figure(figsize=(10,5))
plt.bar(daily.index, daily.values)
plt.title("Road Accidents by Day of Week - 2018")
plt.ylabel("Accidents")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "accidents_by_day.png", dpi=300)
plt.close()

day_hour = pd.crosstab(accident["day_name"], accident["hour"]).reindex(day_order).reindex(columns=range(24), fill_value=0)
plt.figure(figsize=(14,6))
sns.heatmap(day_hour, cmap="viridis")
plt.title("Road Accident Frequency by Day and Hour - 2018")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "accident_day_hour_heatmap.png", dpi=300)
plt.close()

# Motorcycle analysis
vehicle = vehicle.merge(accident[["accident_index","hour","day_name"]], on="accident_index", how="left")
motorbike_map = {2:"125cc and under", 3:"125cc and under", 4:"Over 125cc to 500cc", 5:"Over 500cc"}
vehicle["motorbike_category"] = vehicle["vehicle_type"].map(motorbike_map)
motorbikes = vehicle[vehicle["motorbike_category"].notna()].copy()

motorbike_hour = pd.crosstab(motorbikes["hour"], motorbikes["motorbike_category"]).reindex(range(24), fill_value=0)
motorbike_hour.plot(figsize=(12,6), marker="o")
plt.title("Motorbike Accidents by Hour and Category - 2018")
plt.xlabel("Hour")
plt.ylabel("Accident Records")
plt.xticks(range(24))
plt.tight_layout()
plt.savefig(RESULTS_DIR / "motorbike_accidents_by_hour.png", dpi=300)
plt.close()

# Pedestrian analysis
casualty = casualty.merge(accident[["accident_index","hour","day_name"]], on="accident_index", how="left")
pedestrians = casualty[casualty["casualty_class"] == 3].drop_duplicates("accident_index")

pedestrian_day_hour = pd.crosstab(pedestrians["day_name"], pedestrians["hour"]).reindex(day_order).reindex(columns=range(24), fill_value=0)
plt.figure(figsize=(14,6))
sns.heatmap(pedestrian_day_hour, cmap="YlOrRd")
plt.title("Pedestrian-Involved Accidents by Day and Hour - 2018")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "pedestrian_day_hour_heatmap.png", dpi=300)
plt.close()

print("2018 accidents:", len(accident))
print("Motorcycle records:", len(motorbikes))
print("Unique pedestrian-involved accidents:", len(pedestrians))
print("Figures saved to:", RESULTS_DIR)
