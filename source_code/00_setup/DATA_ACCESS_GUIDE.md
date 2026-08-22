# Data Access Guide — Free Account Registration

Before running the download scripts, you need 3 free accounts.
All data is 100% open access for research use.

---

## 1. NASA Earthdata (for MODIS data)

**Register:** https://urs.earthdata.nasa.gov/users/new  
**Takes:** 2 minutes  
**Used for:** MODIS NDVI, LST, ET, Burned Area  

After registering:
```bash
pip install earthaccess
python -c "import earthaccess; earthaccess.login(strategy='interactive')"
```
This saves credentials to `~/.netrc` permanently — you only do this once.

---

## 2. Copernicus Climate Data Store (for ERA5)

**Register:** https://cds.climate.copernicus.eu/user/register  
**Takes:** 2 minutes  
**Used for:** Precipitation, Temperature, Potential ET, Soil Water  

After registering:
1. Log in and go to: https://cds.climate.copernicus.eu/profile
2. Copy your **UID** and **API Key**
3. Create file `~/.cdsapirc` with this content:
```
url: https://cds.climate.copernicus.eu/api
key: YOUR-UID:YOUR-API-KEY
```
**Example:**
```
url: https://cds.climate.copernicus.eu/api
key: 123456:abcde-fgh-ijklm-nopqr
```

Test it:
```bash
pip install cdsapi
python -c "import cdsapi; c = cdsapi.Client(); print('CDS connected')"
```

---

## 3. CEDA Archive (for ESA CCI Soil Moisture)

**Register:** https://services.ceda.ac.uk/cedasite/register/info/  
**Takes:** 5 minutes  
**Used for:** Daily surface soil moisture  

After registering, you can:
- Download via the automated script (03_download_ESA_CCI_SoilMoisture.py)
- OR browse manually: https://data.ceda.ac.uk/neodc/esacci/soil_moisture/

**Alternative — ESA CCI via Copernicus CDS** (same account as #2):
Search "Soil moisture" on https://cds.climate.copernicus.eu/ — 
ESA CCI SM is available there too with your existing CDS account.

---

## Data Volume Estimates

| Dataset | Study Areas | Est. Size |
|---------|-------------|-----------|
| MODIS MOD13A3 (NDVI/EVI) | Amazon + Sahel + E.Africa | ~40 GB |
| MODIS MOD11A2 (LST) | All 3 areas | ~35 GB |
| MODIS MOD16A2 (ET) | All 3 areas | ~30 GB |
| MODIS MCD64A1 (Fire) | All 3 areas | ~20 GB |
| ERA5 monthly | All 3 areas | ~2 GB |
| ESA CCI SM | All 3 areas | ~15 GB |
| **TOTAL RAW** | | **~142 GB** |
| **Processed (compressed)** | | **~8–12 GB** |

Make sure you have at least **150 GB free disk space** for raw data.
Processed data will be ~10 GB after compression.

---

## Quick Start

Once all 3 accounts are set up:

```bash
# Step 1 — Install all dependencies
pip install -r 00_setup/requirements.txt

# Step 2 — Verify environment
python 00_setup/setup_environment.py

# Step 3 — Run full pipeline
python RUN_PIPELINE.py --phase 1   # Check setup
python RUN_PIPELINE.py --phase 2   # Download data
python RUN_PIPELINE.py --phase 3   # Process data
python RUN_PIPELINE.py --phase 4   # Merge all variables
```
