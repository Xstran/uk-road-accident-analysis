# UK Road Accident Analysis

Analysis of UK road accident data using Python and SQL, combining exploratory analysis, association rule mining, spatial clustering and time-series forecasting.

The project focuses mainly on 2018 road accidents, while 2017–2019 data is used for forecasting. The underlying SQLite database contains accident, vehicle, casualty and geographic records.

## Key Findings

- Accident frequency was lowest overnight and increased sharply around commuting periods, with major peaks around 08:00 and 16:00–17:00.
- Friday recorded the highest accident frequency, while Sunday recorded the lowest.
- Pedestrian-involved accidents were concentrated around 08:00 and 15:00–18:00.
- Severe accidents were more strongly associated with rural areas, higher speed limits, single carriageways and poor lighting.
- West Yorkshire accident locations formed clear spatial clusters rather than being evenly distributed.
- ARIMA forecasts reproduced annual accident totals closely for Metropolitan Police, West Midlands and Kent.
- The West Yorkshire SARIMAX model predicted 37 accidents for July 2018 compared with 33 actual accidents.

## Accident Patterns

![Accidents by hour](results/accidents_by_hour.png)

![Day and hour heatmap](results/accident_day_hour_heatmap.png)

The analysis shows clear morning and late-afternoon peaks in reported accidents, particularly during weekday travel periods.

## Accident Severity

Apriori association-rule mining was used to investigate relationships between accident severity and factors including speed limit, road type, junction type, lighting, weather, road surface and urban/rural location.

![Association rules](results/severity_association_rules.png)

The strongest fatal-accident rule achieved a lift of approximately 2.41, while serious accidents were associated with conditions such as 60 mph roads, single carriageways and darkness without street lighting.

## Spatial Clustering

West Yorkshire accidents were analysed using K-Means and Gaussian Mixture Models.

- K-Means selected 5 clusters using silhouette score.
- Gaussian Mixture Modelling selected 7 components using BIC.
- 4,132 West Yorkshire accident records were analysed.

![K-Means clustering](results/west_yorkshire_kmeans.png)

![GMM clustering](results/west_yorkshire_gmm.png)

## Time-Series Forecasting

Separate ARIMA models were trained on 2017–2018 weekly accident counts and evaluated against 2019 data.

| Police Area | Total Forecast Error |
|---|---:|
| Metropolitan Police | 0.59% |
| West Midlands | 3.94% |
| Kent | 6.24% |

![Metropolitan Police forecast](results/metropolitan_police_forecast.png)

For high-incident West Yorkshire LSOAs, a SARIMAX model was used to forecast daily July 2018 accident counts.

![West Yorkshire SARIMAX forecast](results/west_yorkshire_sarimax_forecast.png)

## Methods

**Data analysis:** SQL, pandas, exploratory data analysis  
**Pattern mining:** Apriori association rules  
**Clustering:** K-Means, Gaussian Mixture Models, silhouette score, BIC  
**Forecasting:** ARIMA, SARIMAX, ADF stationarity testing, AIC model selection, Ljung-Box residual testing  
**Visualisation:** Matplotlib, Seaborn

## Project Structure

```text
uk-road-accident-analysis/
├── data/
│   └── README.md
├── results/
├── scripts/
│   ├── 01_exploratory_analysis.py
│   ├── 02_association_and_clustering.py
│   └── 03_forecasting.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
