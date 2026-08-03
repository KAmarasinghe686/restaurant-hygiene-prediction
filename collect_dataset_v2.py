"""

Restaurant Hygiene Grade Classification Project

Domain      : Public Health Analytics
Target      : Predict inspection grade (A / B / C)
Borough     : Queens only
Date Range  : 2023-01-01 to 2025-12-31

Sources:
  1. NYC Open Data API  - Inspection records
  2. NYC 311 API        - Food complaints per zipcode
  3. NYC Open Data API  - Socioeconomic data by zipcode
     (Census-based: poverty rate, median income)

"""

import requests
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# STEP 1: FETCH RAW INSPECTION DATA — QUEENS ONLY
# Source: NYC Open Data — DOHMH Inspections

def fetch_inspection_data():

    print("STEP 1: Fetching Queens inspection data (2023-2025)")

    url = "https://data.cityofnewyork.us/resource/43nn-pn8j.json"
    params = {
        "$limit": 50000,
        "$where": (
            "inspection_date >= '2023-01-01T00:00:00' "
            "AND inspection_date <= '2025-12-31T00:00:00' "
            "AND score IS NOT NULL "
            "AND boro = 'Queens'"
        ),
        "$order": "inspection_date DESC",
        "$select": (
            "camis,dba,boro,zipcode,cuisine_description,"
            "inspection_date,inspection_type,violation_code,"
            "violation_description,critical_flag,score,grade,action,"
            "latitude,longitude"
        )
    }

    response = requests.get(url, params=params, timeout=60)

    if response.status_code != 200:
        print(f"ERROR: {response.status_code}")
        return None

    df = pd.DataFrame(response.json())

    #Converts score, latitude, longitude to numeric values
    df['score']           = pd.to_numeric(df['score'], errors='coerce')
    df['latitude']        = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude']       = pd.to_numeric(df['longitude'], errors='coerce')
    df['inspection_date'] = pd.to_datetime(df['inspection_date'], errors='coerce')

    df = df.reset_index(drop=True)

    print(f"Total records    : {len(df)}")
    print(f"Columns          : {len(df.columns)}")
    print(f"Date range       : {df['inspection_date'].min()} → {df['inspection_date'].max()}")
    print(f"\nGrade breakdown  :\n{df['grade'].value_counts(dropna=False)}")
    print(f"\nAction breakdown :\n{df['action'].value_counts(dropna=False)}\n")

    return df

# STEP 2: FETCH NYC 311 FOOD COMPLAINTS — BY ZIPCODE
# Source: NYC 311 Service Requests API

def fetch_311_complaints(df):
      
    print("STEP 2: Fetching 311 food complaints by zipcode...")

    url = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"
    params = {
        "$limit": 50000,
        "$where": (
            "complaint_type = 'Food Establishment' "
            "AND created_date >= '2023-01-01T00:00:00' "
            "AND created_date <= '2025-12-31T00:00:00' "
            "AND incident_zip IS NOT NULL"
        ),
        "$select": "incident_zip, created_date"
    }

    resp = requests.get(url, params=params, timeout=60)

    if resp.status_code != 200:
        print(f"  WARNING: 311 API returned {resp.status_code} — skipping")
        df['food_complaints_zipcode'] = None
        return df

    complaints_df = pd.DataFrame(resp.json())

    if complaints_df.empty:
        print("  WARNING: No 311 data returned — skipping")
        df['food_complaints_zipcode'] = None
        return df

    complaints_count = (
        complaints_df
        .groupby('incident_zip')
        .size()
        .reset_index(name='food_complaints_zipcode')
    )
    complaints_count['incident_zip'] = complaints_count['incident_zip'].str.strip()

    df = df.copy()
    df['zipcode'] = df['zipcode'].astype(str).str.strip()

    zipcode_map = dict(zip(
        complaints_count['incident_zip'],
        complaints_count['food_complaints_zipcode']
    ))

    df['food_complaints_zipcode'] = df['zipcode'].map(zipcode_map)

    print(f"Total complaints fetched : {len(complaints_df)}")
    print(f"Matched zipcodes         : {df['food_complaints_zipcode'].notna().sum()}")
    print(f"Unmatched                : {df['food_complaints_zipcode'].isna().sum()}\n")

    return df


# STEP 3: FETCH SOCIOECONOMIC DATA BY ZIPCODE
# Source: NYC Open Data — Census-based poverty & income
# Dataset: American Community Survey (ACS) by NTA/Zipcode

def fetch_socioeconomic_data(df):

    print("STEP 3: Fetching socioeconomic data by zipcode...")

    url = "https://data.cityofnewyork.us/resource/6bgk-pk5e.json"
    params = {
        "$limit": 50000,
        "$select": "zipcode,pct_poverty,median_income_2021"
    }

    resp = requests.get(url, params=params, timeout=30)

    if resp.status_code != 200:
        # fallback — try alternative NYC poverty dataset
        print(f"  First endpoint returned {resp.status_code}, trying fallback...")
        url = "https://data.cityofnewyork.us/resource/ddbm-znfg.json"
        params = {
            "$limit": 50000,
            "$select": "zip_code,percent_poverty,median_household_income"
        }
        resp = requests.get(url, params=params, timeout=30)

    if resp.status_code != 200:
        print(f"  WARNING: Socioeconomic API returned {resp.status_code} — skipping")
        df['pct_poverty']       = None
        df['median_income']     = None
        return df

    socio_df = pd.DataFrame(resp.json())
    print(f"  Socioeconomic records fetched: {len(socio_df)}")
    print(f"  Columns returned: {list(socio_df.columns)}")

    # normalize column names depending on which endpoint worked
    if 'zipcode' in socio_df.columns:
        zip_col = 'zipcode'
    elif 'zip_code' in socio_df.columns:
        zip_col = 'zip_code'
        socio_df = socio_df.rename(columns={'zip_code': 'zipcode'})
    else:
        print("  WARNING: No zipcode column found — skipping")
        df['pct_poverty']   = None
        df['median_income'] = None
        return df

    # normalize value columns
    if 'pct_poverty' not in socio_df.columns and 'percent_poverty' in socio_df.columns:
        socio_df = socio_df.rename(columns={'percent_poverty': 'pct_poverty'})
    if 'median_income_2021' in socio_df.columns:
        socio_df = socio_df.rename(columns={'median_income_2021': 'median_income'})
    elif 'median_household_income' in socio_df.columns:
        socio_df = socio_df.rename(columns={'median_household_income': 'median_income'})

    socio_df['zipcode']      = socio_df['zipcode'].astype(str).str.strip()
    socio_df['pct_poverty']  = pd.to_numeric(socio_df.get('pct_poverty'),  errors='coerce')
    socio_df['median_income']= pd.to_numeric(socio_df.get('median_income'), errors='coerce')

    socio_map_poverty = dict(zip(socio_df['zipcode'], socio_df['pct_poverty']))
    socio_map_income  = dict(zip(socio_df['zipcode'], socio_df['median_income']))

    df = df.copy()
    df['pct_poverty']   = df['zipcode'].map(socio_map_poverty)
    df['median_income'] = df['zipcode'].map(socio_map_income)

    print(f"Poverty matched  : {df['pct_poverty'].notna().sum()}")
    print(f"Income matched   : {df['median_income'].notna().sum()}\n")

    return df

# STEP 4: SAVE RAW DATASET
# No filtering before save

def save_raw_dataset(df, filename='restaurant_hygiene_raw.csv'):

    print("STEP 4: Saving raw dataset...")

    final_cols = [
        # Source 1: NYC Open Data — Inspections
        'camis',
        'dba',
        'boro',
        'zipcode',
        'cuisine_description',
        'inspection_date',
        'inspection_type',
        'violation_code',
        'violation_description',
        'critical_flag',
        'score',
        'action',
        'latitude',
        'longitude',

        # Source 2: NYC 311 API
        'food_complaints_zipcode',

        # Source 3: NYC Socioeconomic (Census via NYC Open Data)
        'pct_poverty',
        'median_income',

        # TARGET
        'grade'
    ]

    final_cols = [c for c in final_cols if c in df.columns]
    df_final   = df[final_cols].copy()

    df_final.to_csv(filename, index=False)

    print(f"File saved       : {filename}")
    print(f"Total rows       : {len(df_final)}")
    print(f"Total columns    : {len(df_final.columns)}")
    print(f"\nAll columns:")
    for i, col in enumerate(df_final.columns, 1):
        print(f"  {i:2}. {col}")
    print(f"\nGrade distribution (raw):")
    print(df_final['grade'].value_counts(dropna=False))
    print(f"\nNull values per column:")
    print(df_final.isnull().sum())

    return df_final


# MAIN

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("  RESTAURANT HYGIENE DATASET COLLECTION")
    print("  Borough    : Queens")
    print("  Date range : 2023 - 2025")
    print("  Sources    : 3 APIs (Inspections, 311, Socioeconomic)")
    print("  Target     : Grade (A / B / C)")
    print("=" * 60 + "\n")

    df_raw        = fetch_inspection_data()

    if df_raw is not None:
        df_complaints = fetch_311_complaints(df_raw)
        df_socio      = fetch_socioeconomic_data(df_complaints)
        df_final      = save_raw_dataset(
                            df_socio,
                            filename='restaurant_hygiene_raw.csv'
                        )

    print("DONE!")
    print("Raw dataset saved to : restaurant_hygiene_raw.csv")
