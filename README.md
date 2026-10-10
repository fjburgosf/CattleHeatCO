# CattleHeatCO

Scripts to compute the exposure of cattle to heat stress in Colombian municipalities,
1981-2025, from hourly ERA5-Land data and the ICA bovine census.

## Contents

`src/download/` downloads ERA5-Land (2 m temperature and dew point; CDS for 1981-2003, NCAR
GDEX for 2004-2026), the ICA census, DANE boundaries, CHIRPS, MapBiomas Colombia, Copernicus
DEM GLO-90, NOAA ONI and IDEAM station data. Each download is logged with its URL, date,
size and SHA-256. `crosscheck_gdex.py` compares one month obtained from both ERA5-Land sources.

`src/livestock/` matches the census to DIVIPOLA codes and flags department-years with
misaligned rows.

`src/preprocessing/` and `src/spatial/` compute the grazing-land fraction, municipal
elevation and the cell-to-municipality weights.

`src/climate/` computes the hourly THI, degree-hours and related annual and monthly
metrics, and the SPI from CHIRPS.

`src/statistics/` runs the trend tests, spatial clusters, exposure decomposition,
compound hot-dry months, sensitivity runs and the comparison with IDEAM stations.

`configs/` holds the name crosswalk, municipal successions, flagged census rows and regions.

## Requirements

Python 3.12 and the packages in `requirements.txt` (or `environment.yml`). ERA5-Land from the
CDS needs an account and a `~/.cdsapirc` file.

## Run order

```
python src/download/download_era5land.py 1981-01 2003-12 --workers 2 --chunk 4
python src/download/download_era5land_gdex.py 2004-01 2026-01 --workers 4
python src/download/download_ica.py
python src/download/download_dem.py
python src/download/download_ideam.py 2020 2025
python src/livestock/build_ica_panel.py
python src/livestock/qc_ica_alignment.py
python src/livestock/finalize_ica.py
python src/preprocessing/mapbiomas_grazing_fraction.py
python src/preprocessing/dem_to_grid.py
python src/spatial/build_weights.py
python src/climate/process_era5land.py 1981 2025
python src/climate/annual_metrics.py 1981 2025
python src/climate/spi_chirps.py
python src/statistics/run_analysis.py
python src/statistics/validate_ideam.py 2020 2025
python src/download/crosscheck_gdex.py
```

Boundaries, CHIRPS, MapBiomas and ONI are single files fetched with
`python src/download/fetch.py <url> <output path> <log name>`. Data are not included.

## Licence

No licence is granted. All rights reserved.
