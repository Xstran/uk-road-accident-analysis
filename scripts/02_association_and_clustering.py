from pathlib import Path
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "accident_data_v1.0.0_2023.db"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

if not DB_PATH.exists():
    raise FileNotFoundError(f"Database not found: {DB_PATH}")

with sqlite3.connect(DB_PATH) as con:
    accident = pd.read_sql_query("SELECT * FROM accident WHERE accident_year = 2018", con)

severity_map = {1:"Fatal", 2:"Serious", 3:"Slight"}
accident["accident_severity_label"] = accident["accident_severity"].map(severity_map)


# APRIORI ASSOCIATION RULES


apriori_df = accident[["accident_severity","speed_limit","road_type","junction_detail","light_conditions",
                       "weather_conditions","road_surface_conditions","urban_or_rural_area"]].copy()

weather_map = {1:"Fine without high winds",2:"Raining without high winds",3:"Snowing without high winds",
               4:"Fine with high winds",5:"Raining with high winds",6:"Snowing with high winds",
               7:"Fog or mist",8:"Other",9:"Unknown"}

surface_map = {1:"Dry",2:"Wet or damp",3:"Snow",4:"Frost or ice",5:"Flood"}
light_map = {1:"Daylight",4:"Darkness street lights lit",5:"Darkness street lights unlit",
             6:"Darkness no street lighting",7:"Darkness street lighting unknown"}
road_map = {1:"Roundabout",2:"One way street",3:"Dual carriageway",6:"Single carriageway",7:"Slip road",9:"Unknown"}
junction_map = {0:"Not at junction",1:"Roundabout",2:"Mini roundabout",3:"T or staggered junction",
                5:"Slip road",6:"Crossroads",7:"More than four arms junction",
                8:"Private drive or entrance",9:"Other junction"}
area_map = {1:"Urban",2:"Rural",3:"Unallocated",-1:"Data missing or out of range"}

apriori_df["Severity"] = "Severity " + apriori_df["accident_severity"].astype(str)
apriori_df["Speed_limit"] = "Speed " + apriori_df["speed_limit"].astype(str) + "mph"
apriori_df["Weather"] = apriori_df["weather_conditions"].map(weather_map).fillna("Unknown weather")
apriori_df["Road_surface"] = apriori_df["road_surface_conditions"].map(surface_map).fillna("Unknown surface")
apriori_df["Light"] = apriori_df["light_conditions"].map(light_map).fillna("Unknown light")
apriori_df["Road_type"] = apriori_df["road_type"].map(road_map).fillna("Unknown road type")
apriori_df["Junction"] = apriori_df["junction_detail"].map(junction_map).fillna("Unknown junction")
apriori_df["Area"] = apriori_df["urban_or_rural_area"].map(area_map).fillna("Unknown area")

selected = apriori_df[["Severity","Speed_limit","Weather","Road_surface","Light","Road_type","Junction","Area"]]
onehot = pd.get_dummies(selected).astype(bool)

itemsets = apriori(onehot, min_support=0.005, use_colnames=True, max_len=3)
rules = association_rules(itemsets, metric="lift", min_threshold=1.0)

rules["antecedents_text"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
rules["consequents_text"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))

severity_rules = rules[(rules["consequents"].apply(len) == 1) &
                       rules["consequents_text"].str.startswith("Severity_")].copy()
severity_rules = severity_rules[~severity_rules["antecedents_text"].str.contains("Unknown|Data missing", regex=True)]
severity_rules = severity_rules.sort_values(["lift","confidence"], ascending=False)

groups = []
for code, name in [(1,"Fatal"),(2,"Serious"),(3,"Slight")]:
    top = severity_rules[severity_rules["consequents_text"] == f"Severity_Severity {code}"].head(5).copy()
    top["severity_name"] = name
    groups.append(top)

top_rules = pd.concat(groups, ignore_index=True)
top_rules["rule_label"] = top_rules["antecedents_text"] + " => " + top_rules["severity_name"]
top_rules = top_rules.sort_values("lift")

plt.figure(figsize=(12,8))
plt.barh(top_rules["rule_label"], top_rules["lift"])
plt.title("Top Accident Severity Association Rules by Lift")
plt.xlabel("Lift")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "severity_association_rules.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# WEST YORKSHIRE SPATIAL CLUSTERING
# ------------------------------------------------------------

west_yorkshire = accident[accident["police_force"] == 13].copy()
west_yorkshire = west_yorkshire.dropna(subset=["location_easting_osgr","location_northing_osgr"])
coordinates = west_yorkshire[["location_easting_osgr","location_northing_osgr"]]

k_results = []
for k in range(2,9):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(coordinates)
    k_results.append({"k":k, "silhouette":silhouette_score(coordinates, labels)})

best_k = max(k_results, key=lambda x: x["silhouette"])["k"]
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
west_yorkshire["kmeans_cluster"] = kmeans.fit_predict(coordinates)

plt.figure(figsize=(9,7))
sns.scatterplot(data=west_yorkshire, x="location_easting_osgr", y="location_northing_osgr",
                hue="kmeans_cluster", s=20, alpha=0.7)
plt.title("K-Means Clusters of West Yorkshire Accidents")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "west_yorkshire_kmeans.png", dpi=300)
plt.close()

gmm_results = []
for n in range(2,9):
    model = GaussianMixture(n_components=n, random_state=42).fit(coordinates)
    gmm_results.append({"components":n, "bic":model.bic(coordinates)})

best_gmm = min(gmm_results, key=lambda x: x["bic"])["components"]
gmm = GaussianMixture(n_components=best_gmm, random_state=42)
west_yorkshire["gmm_cluster"] = gmm.fit_predict(coordinates)

plt.figure(figsize=(9,7))
sns.scatterplot(data=west_yorkshire, x="location_easting_osgr", y="location_northing_osgr",
                hue="gmm_cluster", s=20, alpha=0.7)
plt.title("Gaussian Mixture Clusters of West Yorkshire Accidents")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "west_yorkshire_gmm.png", dpi=300)
plt.close()

print("West Yorkshire accidents:", len(west_yorkshire))
print("Best K-Means clusters:", best_k)
print("Best GMM components:", best_gmm)
print("Figures saved to:", RESULTS_DIR)
