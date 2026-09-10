# Data

This project uses the UK road accident database:

`accident_data_v1.0.0_2023.db`

The database contains road accident records from 2017 to 2020. The analysis focuses mainly on 2018, with 2017 and 2019 also used for forecasting.

Main tables used:

- `accident` — 461,352 records
- `vehicle` — 849,091 records
- `casualty` — 600,332 records
- `lsoa` — 34,378 records

The database is not stored in this repository because of its size.

To run the analysis locally, place the database at:

```text
data/accident_data_v1.0.0_2023.db
