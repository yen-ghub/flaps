"""
Data loading and acquisition for FLAPS.

Consolidates data pipelines from notebooks 1b and 8a:
- BOM FTP weather data download and feature computation
- BITRE flight data download
- Flight + weather + holiday merge into training dataset
- Model and metadata loading with optional Streamlit caching
"""

import glob
import json
import os
import re
import time
from calendar import monthrange
from datetime import date, datetime
from ftplib import FTP
from itertools import permutations

import joblib
import numpy as np
import pandas as pd
import requests
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv

try:
    import holidays as holidays_lib
except ImportError:
    holidays_lib = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_RAW        = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_PROCESSED  = os.path.join(PROJECT_ROOT, 'data', 'processed')
NOWCASTING_MODELS_DIR  = os.path.join(PROJECT_ROOT, 'models', 'nowcasting')
FORECASTING_MODELS_DIR = os.path.join(PROJECT_ROOT, 'models', 'forecasting')

CITY_MAPPING = {
    'Sydney': 'syd',
    'Melbourne': 'mel',
    'Hobart': 'hba',
    'Brisbane': 'bne',
    'Perth': 'per',
    'Adelaide': 'adl',
}

DEFAULT_CITIES = list(CITY_MAPPING.keys())

BITRE_VERSION_RANGE = (9, 16)  # Try versions 9-15 (min, max_exclusive)

# Known filenames for BITRE Monthly Airline Performance data (load factor source).
# The list is checked in order; the first existing file is used.
# Add newer filenames to the front of this list when they become available.
MONTHLY_AIRLINE_PERFORMANCE_CANDIDATES = [
    'monthly-airline-performance-november-2025.xlsx',
    'monthly-airline-performance-october-2025.xlsx',
    'monthly-airline-performance-september-2025.xlsx',
    'monthly-airline-performance-august-2025.xlsx',
]
# Range chosen to cover ~6 months of potential releases
# Adjust if BITRE releases more/less frequently or skips version numbers

# BOM FTP airport weather station paths (from notebook 1b)
AIRPORT_FTP_PATHS = {
    'adl': '/anon/gen/clim_data/IDCKWCDEA0/tables/sa/adelaide_airport',
    'bne': '/anon/gen/clim_data/IDCKWCDEA0/tables/qld/brisbane_aero',
    'syd': '/anon/gen/clim_data/IDCKWCDEA0/tables/nsw/sydney_airport_amo',
    'per': '/anon/gen/clim_data/IDCKWCDEA0/tables/wa/perth_airport',
    'hba': '/anon/gen/clim_data/IDCKWCDEA0/tables/tas/hobart_airport',
    'mel': '/anon/gen/clim_data/IDCKWCDEA0/tables/vic/melbourne_airport',
}

STATES = ['ACT', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA']

# Flightera API
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

FLIGHTERA_BASE_URL = 'https://flightera-flight-data.p.rapidapi.com'
FLIGHTERA_REQUEST_DELAY = 0.1   # seconds between API calls
FLIGHTERA_REQUEST_TIMEOUT = 30
FLIGHTERA_MAX_RETRIES = 3
FLIGHTERA_MIN_FLIGHTS = 50

CITY_TO_ICAO = {
    'Sydney': 'YSSY', 'Melbourne': 'YMML', 'Brisbane': 'YBBN',
    'Perth': 'YPPH', 'Adelaide': 'YPAD', 'Hobart': 'YMHB',
}

FLIGHTERA_AIRLINES = [
    {'bitre_name': 'Qantas',                            'iata': 'QF', 'icao': 'QFA'},
    {'bitre_name': 'Virgin Australia',                   'iata': 'VA', 'icao': 'VAA'},
    {'bitre_name': 'Jetstar',                            'iata': 'JQ', 'icao': 'JST'},
    {'bitre_name': 'QantasLink',                         'iata': 'QF', 'icao': 'QLK'},
    {'bitre_name': 'Rex Airlines',                       'iata': 'ZL', 'icao': 'RXA'},
    {'bitre_name': 'Regional Express',                   'iata': 'ZL', 'icao': 'RXA'},
    {'bitre_name': 'Tigerair Australia',                 'iata': 'TT', 'icao': 'TGW'},
    {'bitre_name': 'Virgin Australia Regional Airlines', 'iata': 'VA', 'icao': 'VOZ'},
]


# ===========================================================================
# 1. BOM Weather Data Download (from notebook 1b)
# ===========================================================================

def download_bom_weather_data(cities=None, output_base_path=None):
    """
    Download weather CSV files from BOM FTP for specified cities.

    Parameters
    ----------
    cities : list of str
        City codes (e.g. ['syd', 'mel']). Defaults to all cities.
    output_base_path : str
        Base directory for saving. Defaults to data/raw/.

    Returns
    -------
    dict : {city_code: number_of_files_downloaded}
    """
    if cities is None:
        cities = list(AIRPORT_FTP_PATHS.keys())
    if output_base_path is None:
        output_base_path = DATA_RAW

    ftp_host = 'ftp.bom.gov.au'
    download_summary = {}

    print(f"Connecting to {ftp_host}...")
    ftp = FTP(ftp_host)
    ftp.login(user='anonymous', passwd='guest@example.com')
    print("Connected successfully.\n")

    for city in cities:
        city = city.lower()
        if city not in AIRPORT_FTP_PATHS:
            print(f"Warning: Unknown city '{city}'. Skipping.")
            continue

        ftp_path = AIRPORT_FTP_PATHS[city]
        output_folder = os.path.join(output_base_path, city)
        os.makedirs(output_folder, exist_ok=True)

        print(f"Downloading data for {city.upper()}...")
        ftp.cwd(ftp_path)
        files = [f for f in ftp.nlst() if f.endswith('.csv')]
        print(f"  Found {len(files)} CSV files on server")

        # Determine the most-recent month so we always re-download it
        # (BOM updates the current month's file daily as observations arrive)
        def _extract_yyyymm(fname):
            m = re.search(r'-(\d{6})\.csv$', fname)
            return m.group(1) if m else ''

        latest_yyyymm = max((_extract_yyyymm(f) for f in files), default='')

        downloaded = 0
        skipped = 0
        for filename in files:
            local_path = os.path.join(output_folder, filename)
            is_latest = (_extract_yyyymm(filename) == latest_yyyymm)
            if os.path.exists(local_path) and not is_latest:
                skipped += 1
                continue
            with open(local_path, 'wb') as f:
                ftp.retrbinary(f'RETR {filename}', f.write)
            downloaded += 1
            if downloaded % 50 == 0:
                print(f"    Downloaded {downloaded} files...")

        download_summary[city] = downloaded
        print(f"  Completed: {downloaded} downloaded, {skipped} skipped\n")

    ftp.quit()
    print("Download complete.")
    return download_summary


# ===========================================================================
# 2. Weather Feature Computation (from notebook 1b)
# ===========================================================================

def load_clean_weather_csv(file_path):
    """Load and clean a single BOM weather CSV file."""
    df = pd.read_csv(
        file_path, skiprows=range(1, 13), skipfooter=1,
        encoding='latin-1', engine='python'
    )
    df.columns = [
        'station', 'date', 'evapotranspiration', 'rain', 'panevaporation',
        'temp_max', 'temp_min', 'hum_max', 'hum_min', 'wind_speed', 'solar_rad'
    ]
    df = df.drop('station', axis=1)
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')

    numeric_cols = [
        'evapotranspiration', 'rain', 'panevaporation',
        'temp_max', 'temp_min', 'hum_max', 'hum_min', 'wind_speed', 'solar_rad'
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['year_month'] = df['date'].dt.to_period('M')
    return df


def compute_weather_features(city_code, input_base_path=None, output_path=None):
    """
    Process all weather CSVs for a city into monthly features.

    Parameters
    ----------
    city_code : str
        e.g. 'syd', 'mel'
    input_base_path : str
        Base directory containing city folders. Defaults to data/raw/.
    output_path : str
        Path for output CSV. Defaults to data/processed/features_{city}.csv.

    Returns
    -------
    DataFrame : monthly weather features
    """
    if input_base_path is None:
        input_base_path = DATA_RAW
    if output_path is None:
        output_path = os.path.join(DATA_PROCESSED, f'features_{city_code}.csv')

    input_folder = os.path.join(input_base_path, city_code)
    files = sorted(glob.glob(os.path.join(input_folder, '*.csv')))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {input_folder}")

    print(f"Processing {city_code.upper()}: {len(files)} files...")
    dfs = [load_clean_weather_csv(f) for f in files]
    df_combined = pd.concat(dfs, ignore_index=True)
    print(f"  {len(df_combined)} daily records, "
          f"{df_combined['date'].min().date()} to {df_combined['date'].max().date()}")

    # Basic monthly statistics
    monthly_stats = df_combined.groupby('year_month').agg({
        'date': 'count',
        'temp_max': ['max', 'mean'],
        'temp_min': ['min', 'mean'],
        'rain': ['max', 'mean'],
        'wind_speed': ['mean', 'max'],
        'hum_max': 'mean',
        'hum_min': 'mean',
    }).reset_index()

    # Flatten multi-level columns
    cols = []
    for c in monthly_stats.columns:
        cols.append(c[0] if c[1] == '' else '_'.join(c))
    monthly_stats.columns = cols

    monthly_stats.rename(columns={
        'date_count': 'days_in_month',
        'rain_mean': 'avg_rainfall_per_day',
        'temp_max_max': 'max_temperature',
        'temp_max_mean': 'avg_max_temp',
        'temp_min_min': 'min_temperature',
        'temp_min_mean': 'avg_min_temp',
        'rain_max': 'max_daily_rainfall',
        'wind_speed_mean': 'avg_wind_speed',
        'wind_speed_max': 'max_wind_speed',
        'hum_max_mean': 'avg_max_humidity',
        'hum_min_mean': 'avg_min_humidity',
    }, inplace=True)

    # Temperature features
    temp_features = df_combined.groupby('year_month').apply(
        lambda x: pd.Series({
            'temp_range_mean': (x['temp_max'] - x['temp_min']).mean(),
            'days_above_35C': (x['temp_max'] > 35).sum(),
            'temp_volatility': x['temp_max'].diff().abs().mean(),
        }), include_groups=False
    ).reset_index()
    monthly_stats = monthly_stats.merge(temp_features, on='year_month', how='left')

    # Precipitation features
    precip_features = df_combined.groupby('year_month').apply(
        lambda x: pd.Series({
            'rainy_days': (x['rain'] > 0).sum(),
            'heavy_rain_days': (x['rain'] > 10).sum(),
            'avg_rainfall_on_rainy_days': (
                x.loc[x['rain'] > 0, 'rain'].mean() if (x['rain'] > 0).any() else 0
            ),
        }), include_groups=False
    ).reset_index()
    monthly_stats = monthly_stats.merge(precip_features, on='year_month', how='left')

    # Wind features
    wind_features = df_combined.groupby('year_month').apply(
        lambda x: pd.Series({
            'days_high_wind': (x['wind_speed'] > 8).sum(),
            'wind_speed_std': x['wind_speed'].std(),
        }), include_groups=False
    ).reset_index()
    monthly_stats = monthly_stats.merge(wind_features, on='year_month', how='left')

    # Humidity features
    humidity_features = df_combined.groupby('year_month').apply(
        lambda x: pd.Series({
            'days_high_humidity': (x['hum_max'] > 90).sum(),
        }), include_groups=False
    ).reset_index()
    monthly_stats = monthly_stats.merge(humidity_features, on='year_month', how='left')

    # Extreme weather composite
    extreme_features = df_combined.groupby('year_month').apply(
        lambda x: pd.Series({
            'extreme_weather_days': (
                (x['temp_max'] > 35) | (x['rain'] > 10) |
                (x['wind_speed'] > 8) | (x['hum_max'] > 95)
            ).sum(),
        }), include_groups=False
    ).reset_index()
    monthly_stats = monthly_stats.merge(extreme_features, on='year_month', how='left')

    # Convert year_month Period to string for CSV compatibility
    monthly_stats['year_month'] = monthly_stats['year_month'].astype(str)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    monthly_stats.to_csv(output_path, index=False)
    print(f"  Saved {len(monthly_stats)} months to {output_path}")
    return monthly_stats


# ===========================================================================
# 3. BITRE Flight Data Download (from notebook 8a)
# ===========================================================================

def download_bitre_data(save_path=None):
    """
    Download the latest BITRE flight data by trying recent months.

    Tries both new version-numbered pattern (otp_time_series_master_current_N.xlsx)
    and old date-based pattern (OTP_Time_Series_Master_Current_month_year.xlsx).

    Returns
    -------
    str or None : path to downloaded file, or None if download failed
    """
    if save_path is None:
        save_path = DATA_RAW

    base_url = "https://www.bitre.gov.au/sites/default/files/documents/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }

    # Try new version-numbered pattern first (newest to oldest)
    min_version, max_version = BITRE_VERSION_RANGE
    # for version in range(max_version - 1, min_version - 1, -1):
    #     filename = f"otp_time_series_master_current_{version}.xlsx"
    #     url = base_url + filename

    #     print(f"Attempting to download: {filename}")
    #     try:
    #         response = requests.get(url, headers=headers, timeout=60)
    #         if response.status_code == 200:
    #             filepath = os.path.join(save_path, filename)
    #             with open(filepath, 'wb') as f:
    #                 f.write(response.content)
    #             print(f"  Successfully downloaded: {filename}")
    #             return filepath
    #         else:
    #             print(f"  Not found (HTTP {response.status_code})")
    #     except requests.RequestException as e:
    #         print(f"  Failed: {type(e).__name__}")

    # Fall back to old date-based pattern
    current_date = datetime.now()
    for months_back in range(1, 4):
        target_date = current_date - relativedelta(months=months_back)
        month_name = target_date.strftime("%B").lower()
        year = target_date.year
        filename = f"OTP_Time_Series_Master_Current_{month_name}_{year}.xlsx"

        url = base_url + filename

        print(f"Attempting to download: {filename}")
        try:
            response = requests.get(url, headers=headers, timeout=60)
            if response.status_code == 200:
                filepath = os.path.join(save_path, filename)
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                print(f"  Successfully downloaded: {filename}")
                return filepath
            else:
                print(f"  Not found (HTTP {response.status_code})")
        except requests.RequestException as e:
            print(f"  Failed: {type(e).__name__}")

    print("Automatic download failed. Will use existing local file.")
    return None


def _get_latest_bitre_file():
    """
    Find the most recent local BITRE Excel file.

    Searches for both old pattern (OTP_Time_Series_Master_Current_*.xlsx)
    and new pattern (otp_time_series_master_current_*.xlsx).

    Returns the most recently modified file.
    """
    # Search for both old and new patterns
    pattern_old = os.path.join(DATA_RAW, 'OTP_Time_Series_Master_Current_*.xlsx')
    #pattern_new = os.path.join(DATA_RAW, 'otp_time_series_master_current_*.xlsx')

    local_files = glob.glob(pattern_old)
    #local_files = glob.glob(pattern_old) + glob.glob(pattern_new)

    if not local_files:
        raise FileNotFoundError("No BITRE data file found in data/raw/")

    # Return the most recently modified file
    return max(local_files, key=os.path.getmtime)


# ===========================================================================
# 4. Holiday Features (from notebook 8a)
# ===========================================================================

def _get_school_holiday_periods(year):
    """Approximate Australian school holiday periods."""
    return [
        (date(year - 1, 12, 18), date(year, 1, 28)),
        (date(year, 4, 8), date(year, 4, 23)),
        (date(year, 6, 27), date(year, 7, 14)),
        (date(year, 9, 23), date(year, 10, 7)),
        (date(year, 12, 18), date(year, 12, 31)),
    ]


def _count_school_holiday_days(year, month):
    """Count school holiday days in a given month."""
    periods = _get_school_holiday_periods(year)
    if month == 1:
        periods.extend(_get_school_holiday_periods(year))
    _, days_in_month = monthrange(year, month)
    count = 0
    for day in range(1, days_in_month + 1):
        current = date(year, month, day)
        for start, end in periods:
            if start <= current <= end:
                count += 1
                break
    return count


def compute_holiday_features(years):
    """
    Compute monthly holiday features for a range of years.

    Returns DataFrame with columns: year, month_num, n_public_holidays_total, pct_school_holiday
    """
    if holidays_lib is None:
        raise ImportError("'holidays' package is required for holiday features")

    state_holidays = {
        state: holidays_lib.Australia(years=years, prov=state) for state in STATES
    }

    rows = []
    for year in years:
        for month in range(1, 13):
            _, days_in_month = monthrange(year, month)

            # Unique public holiday dates across all states
            all_dates = set()
            for state in STATES:
                for day in range(1, days_in_month + 1):
                    d = date(year, month, day)
                    if d in state_holidays[state]:
                        all_dates.add(d)

            school_days = _count_school_holiday_days(year, month)
            rows.append({
                'year': year,
                'month_num': month,
                'n_public_holidays_total': len(all_dates),
                'pct_school_holiday': round(school_days / days_in_month, 4),
            })

    return pd.DataFrame(rows)


# ===========================================================================
# 5. Prepare Training Data (from notebook 8a)
# ===========================================================================

def prepare_training_data(cities=None, bitre_file=None, output_filename=None):
    """
    Full pipeline: load BITRE + weather + holidays, merge, save.

    Parameters
    ----------
    cities : list of str
        City full names (e.g. ['Sydney', 'Melbourne']). Defaults to all 6.
    bitre_file : str or None
        Path to BITRE Excel file. None = use latest local file.
    output_filename : str
        Output CSV filename. Defaults to 'ml_training_data_multiroute_hols.csv'.

    Returns
    -------
    DataFrame : merged training data
    """
    if cities is None:
        cities = DEFAULT_CITIES
    if output_filename is None:
        output_filename = 'ml_training_data_multiroute_hols.csv'

    # --- Load BITRE data ---
    if bitre_file is None:
        bitre_file = _get_latest_bitre_file()
    print(f"Loading BITRE data from: {bitre_file}")

    all_sheets = pd.read_excel(bitre_file, sheet_name=None)
    df_raw = pd.concat(all_sheets.values(), ignore_index=True)
    df_raw = df_raw.drop_duplicates()
    df_raw['Month'] = pd.to_datetime(df_raw['Month'], format='%m/%Y')
    df_raw['year_month'] = df_raw['Month'].dt.to_period('M').astype(str)

    # --- Filter to route pairs ---
    route_pairs = list(permutations(cities, 2))
    route_names = [f"{dep}-{arr}" for dep, arr in route_pairs]

    df_flights = df_raw[
        (df_raw['Route'].isin(route_names)) &
        (df_raw['Airline'] != 'All Airlines')
    ].copy()

    # Clean
    df_flights = df_flights[df_flights['Sectors Flown'] > 0].copy()
    df_flights['delay_rate'] = (
        (df_flights['Sectors Flown'] - df_flights['Arrivals On Time']) / df_flights['Sectors Flown']
    )
    df_flights['is_high_delay'] = (df_flights['delay_rate'] > 0.25).astype(int)
    df_flights['year'] = df_flights['Month'].dt.year

    # Exclude COVID period
    covid_mask = (df_flights['Month'] >= '2020-04-01') & (df_flights['Month'] <= '2020-12-31')
    df_flights = df_flights[~covid_mask].copy()

    # Rename columns
    column_mapping = {
        'Route': 'route',
        'Departing Port': 'departing_port',
        'Arriving Port': 'arriving_port',
        'Airline': 'airline',
        'Month': 'month',
        'Sectors Scheduled': 'sectors_scheduled',
        'Sectors Flown': 'sectors_flown',
        'Cancellations': 'cancellations',
        'Arrivals On Time': 'arrivals_on_time',
        'Arrivals Delayed': 'arrivals_delayed',
        'Cancellations \n\n(%)': 'cancellations_pct',
    }
    df_flights = df_flights.rename(columns=column_mapping)
    print(f"Flight records after cleaning: {len(df_flights)}")

    # --- Load and merge weather ---
    def _prepare_weather(city, suffix):
        code = CITY_MAPPING[city]
        path = os.path.join(DATA_PROCESSED, f'features_{code}.csv')
        df = pd.read_csv(path)
        rename = {col: f"{col}{suffix}" for col in df.columns if col != 'year_month'}
        return df.rename(columns=rename)

    weather_dep = {city: _prepare_weather(city, '_dep') for city in cities}
    weather_arr = {city: _prepare_weather(city, '_arr') for city in cities}

    merged = []
    for dep_city, arr_city in route_pairs:
        mask = (
            (df_flights['departing_port'] == dep_city) &
            (df_flights['arriving_port'] == arr_city)
        )
        df_route = df_flights[mask].copy()
        if len(df_route) == 0:
            continue
        df_route = df_route.merge(weather_dep[dep_city], on='year_month', how='left')
        df_route = df_route.merge(weather_arr[arr_city], on='year_month', how='left')
        merged.append(df_route)

    df_combined = pd.concat(merged, ignore_index=True)
    print(f"Merged records: {len(df_combined)}")

    # --- Rearrange columns ---
    flight_cols = [
        'departing_port', 'arriving_port', 'airline', 'month', 'year_month', 'year',
        'sectors_scheduled', 'sectors_flown', 'arrivals_on_time', 'arrivals_delayed',
        'cancellations', 'cancellations_pct',
    ]
    target_cols = ['delay_rate', 'is_high_delay']
    weather_dep_cols = sorted([c for c in df_combined.columns if c.endswith('_dep')])
    weather_arr_cols = sorted([c for c in df_combined.columns if c.endswith('_arr')])

    flight_cols = [c for c in flight_cols if c in df_combined.columns]
    column_order = flight_cols + target_cols + weather_dep_cols + weather_arr_cols
    df_final = df_combined[column_order].copy()
    df_final = df_final.sort_values(
        ['year_month', 'airline', 'departing_port', 'arriving_port']
    ).reset_index(drop=True)

    # --- Add holiday features ---
    df_final['year_month_dt'] = pd.to_datetime(df_final['year_month'])
    df_final['month_num'] = df_final['year_month_dt'].dt.month

    year_range = list(range(
        int(df_final['year'].min()),
        int(df_final['year'].max()) + 1
    ))
    df_holidays = compute_holiday_features(year_range)
    df_final['year'] = df_final['year'].astype(int)
    df_final = df_final.merge(df_holidays, on=['year', 'month_num'], how='left')

    # --- Save ---
    output_path = os.path.join(DATA_PROCESSED, output_filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False)
    print(f"Training data saved to: {output_path} ({len(df_final)} rows)")
    return df_final


# ===========================================================================
# 6. Flightera API Pipeline (supplements BITRE with near-real-time data)
# ===========================================================================

def _flightera_api_get(endpoint, params, headers):
    """GET request to Flightera API with retry and rate limiting."""
    url = f'{FLIGHTERA_BASE_URL}{endpoint}'
    delay = 1.0
    for attempt in range(FLIGHTERA_MAX_RETRIES):
        try:
            resp = requests.get(
                url, headers=headers, params=params,
                timeout=FLIGHTERA_REQUEST_TIMEOUT,
            )
        except requests.RequestException:
            if attempt == FLIGHTERA_MAX_RETRIES - 1:
                return None
            time.sleep(delay)
            delay *= 2
            continue

        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 429 or resp.status_code >= 500:
            if attempt == FLIGHTERA_MAX_RETRIES - 1:
                return None
            time.sleep(delay)
            delay *= 2
            continue
        # 400/404 — no data for this query, don't retry
        return None
    return None


def _discover_flight_numbers(routes_df, airlines, year_month, headers):
    """Discover active flight numbers for each airline-route via /airline/routes."""
    year, month = year_month.split('-')
    discovery_dates = [f'{year}-{month}-{d:02d}' for d in range(8, 15)]

    discovery_rows = []
    seen = set()
    total_routes = len(routes_df)

    for route_idx, (_, route) in enumerate(routes_df.iterrows(), 1):
        dep, arr = route['departing_port'], route['arriving_port']
        print(f"  Route {route_idx}/{total_routes}: {dep} -> {arr}")

        for airline in airlines:
            for dt in discovery_dates:
                time.sleep(FLIGHTERA_REQUEST_DELAY)
                result = _flightera_api_get('/airline/routes', {
                    'departure': route['origin_icao'],
                    'arrival': route['dest_icao'],
                    'date': dt,
                    'airline': airline['icao'],
                }, headers)

                routes_list = _extract_flnrs(result)

                if len(routes_list) == 0:
                    # Fallback to IATA code
                    time.sleep(FLIGHTERA_REQUEST_DELAY)
                    result = _flightera_api_get('/airline/routes', {
                        'departure': route['origin_icao'],
                        'arrival': route['dest_icao'],
                        'date': dt,
                        'airline': airline['iata'],
                    }, headers)
                    routes_list = _extract_flnrs(result)

                for rec in routes_list:
                    flnr = rec.get('flnr', '')
                    key = (dep, arr, airline['icao'], flnr)
                    if flnr and key not in seen:
                        seen.add(key)
                        discovery_rows.append({
                            'departing_port': dep,
                            'arriving_port': arr,
                            'origin_icao': route['origin_icao'],
                            'dest_icao': route['dest_icao'],
                            'airline_icao': airline['icao'],
                            'bitre_name': airline['bitre_name'],
                            'flnr': flnr,
                        })

    print(f"  Unique flight numbers discovered: {len(discovery_rows)}")
    return pd.DataFrame(discovery_rows)


def _extract_flnrs(result):
    """Extract flnr list from Flightera API response."""
    if isinstance(result, dict) and 'routes' in result:
        return result['routes']
    if isinstance(result, list):
        return result
    return []


def _fetch_monthly_stats(df_flnr, year, month, headers):
    """Fetch pre-aggregated monthly stats for each discovered flight number."""
    unique_flnrs = df_flnr['flnr'].unique()
    total = len(unique_flnrs)
    print(f"  Fetching stats for {total} flight numbers...")

    stats_rows = []
    for i, flnr in enumerate(unique_flnrs, 1):
        if i % 50 == 0:
            print(f"    Flight number {i}/{total}")
        time.sleep(FLIGHTERA_REQUEST_DELAY)
        result = _flightera_api_get('/flight/statistics/monthly', {
            'flnr': flnr,
            'month': month,
            'year': year,
        }, headers)

        if result is None or not isinstance(result, list):
            continue
        for rec in result:
            rec['_flnr_query'] = flnr
            stats_rows.append(rec)

    print(f"  Stats records collected: {len(stats_rows)}")
    return pd.DataFrame(stats_rows) if stats_rows else pd.DataFrame()


def _aggregate_to_bitre_format(df_stats, df_flnr, year_month):
    """Aggregate per-flight stats to airline-route level in BITRE-compatible format."""
    year, month = year_month.split('-')

    df_joined = df_stats.merge(
        df_flnr[['flnr', 'departing_port', 'arriving_port', 'origin_icao', 'dest_icao', 'bitre_name']],
        left_on='_flnr_query', right_on='flnr',
        how='left', suffixes=('', '_disc'),
    )
    df_joined['count'] = pd.to_numeric(df_joined['count'], errors='coerce').fillna(0).astype(int)
    df_joined['delay_percentage'] = pd.to_numeric(df_joined['delay_percentage'], errors='coerce').fillna(0)
    df_joined['cancelled_percentage'] = pd.to_numeric(df_joined['cancelled_percentage'], errors='coerce').fillna(0)
    df_joined = df_joined[df_joined['count'] > 0].copy()

    # Keep only the stats row for the specific route leg — the stats API returns all
    # segments a flight number operates globally, not just the queried origin→destination.
    df_joined = df_joined[
        (df_joined['from'] == df_joined['origin_icao']) &
        (df_joined['to'] == df_joined['dest_icao'])
    ].copy()

    # Reclassify by callsign: QLK callsigns are QantasLink operated under Qantas flight numbers
    qlk_mask = df_joined['callsign'].str.startswith('QLK', na=False)
    df_joined.loc[qlk_mask, 'bitre_name'] = 'QantasLink'

    # Impute delay_percentage for rows with missing avg_delay (no timing data recorded).
    # These rows have delay_percentage = 0 by default, biasing the aggregate downward.
    # Replace with the flight-count-weighted mean of the other flights in the same airline group.
    missing_mask = df_joined['avg_delay'].isna()
    for airline, grp_idx in df_joined.groupby('bitre_name').groups.items():
        grp = df_joined.loc[grp_idx]
        known = grp[~missing_mask.loc[grp_idx]]
        if len(known) == 0:
            continue
        mean_delay_pct = (known['delay_percentage'] * known['count']).sum() / known['count'].sum() + 10
        df_joined.loc[grp_idx[missing_mask.loc[grp_idx]], 'delay_percentage'] = mean_delay_pct

    # Scale count and delay_percentage to flown flights only (exclude cancellations).
    # Flightera count = scheduled; delay_percentage = delayed / scheduled.
    # BITRE sectors_flown = scheduled * (1 - cancel_rate); delay_rate = delayed / flown.
    cancel_rate = df_joined['cancelled_percentage'] / 100.0
    df_joined['flown'] = df_joined['count'] * (1 - cancel_rate)
    df_joined['delay_pct_flown'] = df_joined['delay_percentage'] / (1 - cancel_rate).clip(lower=1e-6)

    agg_rows = []
    for (dep, arr, airline), grp in df_joined.groupby(['departing_port', 'arriving_port', 'bitre_name']):
        sectors_scheduled = int(grp['count'].sum())
        sectors_flown = int(grp['flown'].sum().round())
        cancellations = sectors_scheduled - sectors_flown
        cancel_pct = round(cancellations / sectors_scheduled * 100, 2) if sectors_scheduled > 0 else 0
        w = grp['flown']
        delay_rate = (grp['delay_pct_flown'] * w).sum() / w.sum() / 100.0 if w.sum() > 0 else 0
        arrivals_delayed = round(sectors_flown * delay_rate)
        arrivals_on_time = sectors_flown - arrivals_delayed

        agg_rows.append({
            'departing_port': dep,
            'arriving_port': arr,
            'airline': airline,
            'month': pd.Timestamp(f'{year}-{month}-01'),
            'year_month': year_month,
            'year': int(year),
            'sectors_scheduled': sectors_scheduled,
            'sectors_flown': sectors_flown,
            'arrivals_on_time': arrivals_on_time,
            'arrivals_delayed': arrivals_delayed,
            'cancellations': cancellations,
            'cancellations_pct': cancel_pct,
            'delay_rate': round(delay_rate, 6),
            'is_high_delay': int(delay_rate > 0.25),
        })

    df_bitre = pd.DataFrame(agg_rows)
    df_filtered = df_bitre[df_bitre['sectors_flown'] >= FLIGHTERA_MIN_FLIGHTS].copy()
    print(f"  Airline-route groups: {len(df_bitre)} total, {len(df_filtered)} with >= {FLIGHTERA_MIN_FLIGHTS} flights")
    return df_filtered


def append_flightera_data(year_month, cities=None):
    """
    Fetch flight data for a given month via Flightera API and merge into training data.

    Discovers flight numbers, fetches monthly stats, aggregates to BITRE format,
    merges weather + holiday features, and appends to the training CSV
    (replacing any existing rows for that month).

    Parameters
    ----------
    year_month : str
        Target month in 'YYYY-MM' format.
    cities : list of str or None
        City full names. Defaults to all 6.
    """
    if cities is None:
        cities = DEFAULT_CITIES

    rapidapi_key = os.environ.get('RAPIDAPI_KEY', '')
    if not rapidapi_key:
        raise ValueError('RAPIDAPI_KEY not set in environment or .env file.')

    headers = {
        'x-rapidapi-host': 'flightera-flight-data.p.rapidapi.com',
        'x-rapidapi-key': rapidapi_key,
    }

    year, month = year_month.split('-')

    # Build routes DataFrame
    route_pairs = list(permutations(cities, 2))
    routes_df = pd.DataFrame(route_pairs, columns=['departing_port', 'arriving_port'])
    routes_df['origin_icao'] = routes_df['departing_port'].map(CITY_TO_ICAO)
    routes_df['dest_icao'] = routes_df['arriving_port'].map(CITY_TO_ICAO)

    # Deduplicate airlines by ICAO
    airlines = []
    seen_icao = set()
    for a in FLIGHTERA_AIRLINES:
        if a['icao'] not in seen_icao:
            seen_icao.add(a['icao'])
            airlines.append(a)

    # Step 1: Discover flight numbers
    print(f"Discovering flight numbers for {year_month}...")
    df_flnr = _discover_flight_numbers(routes_df, airlines, year_month, headers)
    if len(df_flnr) == 0:
        print("No flight numbers discovered. Skipping Flightera update.")
        return

    # Step 2: Fetch monthly stats
    print(f"Fetching monthly statistics...")
    df_stats = _fetch_monthly_stats(df_flnr, year, month, headers)
    if len(df_stats) == 0:
        print("No statistics returned. Skipping Flightera update.")
        return

    # Step 3: Aggregate to BITRE format
    print(f"Aggregating to BITRE-compatible format...")
    df_bitre = _aggregate_to_bitre_format(df_stats, df_flnr, year_month)
    if len(df_bitre) == 0:
        print("No airline-routes with enough flights. Skipping Flightera update.")
        return

    # Step 4: Merge weather + holidays and append to training data
    print(f"Merging with weather and holiday features...")
    training_path = os.path.join(DATA_PROCESSED, 'ml_training_data_multiroute_hols.csv')
    df_train = pd.read_csv(training_path)

    existing = (df_train['year_month'] == year_month).sum()
    if existing:
        print(f"  Replacing {existing} existing rows for {year_month}")
    df_train_clean = df_train[df_train['year_month'] != year_month].copy()

    # Merge weather features
    def _prepare_weather(city, suffix):
        code = CITY_MAPPING[city]
        path = os.path.join(DATA_PROCESSED, f'features_{code}.csv')
        df = pd.read_csv(path)
        rename = {col: f"{col}{suffix}" for col in df.columns if col != 'year_month'}
        return df.rename(columns=rename)

    weather_dep = {city: _prepare_weather(city, '_dep') for city in cities}
    weather_arr = {city: _prepare_weather(city, '_arr') for city in cities}

    merged_parts = []
    for dep_city, arr_city in permutations(cities, 2):
        mask = (df_bitre['departing_port'] == dep_city) & (df_bitre['arriving_port'] == arr_city)
        df_route = df_bitre[mask].copy()
        if len(df_route) == 0:
            continue
        df_route = df_route.merge(weather_dep[dep_city], on='year_month', how='left')
        df_route = df_route.merge(weather_arr[arr_city], on='year_month', how='left')
        merged_parts.append(df_route)

    if merged_parts:
        df_new = pd.concat(merged_parts, ignore_index=True)
    else:
        df_new = df_bitre.copy()

    # Add holiday features
    df_new['year_month_dt'] = pd.to_datetime(df_new['year_month'])
    df_new['month_num'] = df_new['year_month_dt'].dt.month
    df_new['year'] = df_new['year'].astype(int)
    df_holidays = compute_holiday_features([int(year)])
    df_new = df_new.merge(df_holidays, on=['year', 'month_num'], how='left')

    # Align columns and append
    for col in df_train_clean.columns:
        if col not in df_new.columns:
            df_new[col] = np.nan
    df_new = df_new[df_train_clean.columns]

    df_combined = pd.concat([df_train_clean, df_new], ignore_index=True)
    df_combined = df_combined.sort_values(
        ['year_month', 'airline', 'departing_port', 'arriving_port']
    ).reset_index(drop=True)

    df_combined.to_csv(training_path, index=False)
    print(f"  Training data updated: {len(df_combined)} rows ({len(df_new)} new for {year_month})")


# ===========================================================================
# 7. Full Data Update Pipeline
# ===========================================================================

def update_all_data(cities=None):
    """
    Orchestrate full data refresh: download weather + flights, compute features, merge.

    Parameters
    ----------
    cities : list of str
        City full names. Defaults to all 6.
    """
    if cities is None:
        cities = DEFAULT_CITIES

    city_codes = [CITY_MAPPING[c] for c in cities]

    # Step 1: Download weather data from BOM FTP
    print("=" * 80)
    print("STEP 1: Downloading BOM weather data...")
    print("=" * 80)
    download_bom_weather_data(city_codes)

    # Step 2: Compute weather features
    print("\n" + "=" * 80)
    print("STEP 2: Computing weather features...")
    print("=" * 80)
    for code in city_codes:
        compute_weather_features(code)

    # Step 3: Download BITRE flight data
    print("\n" + "=" * 80)
    print("STEP 3: Downloading BITRE flight data...")
    print("=" * 80)
    bitre_file = download_bitre_data()

    # Step 4: Merge into training data
    print("\n" + "=" * 80)
    print("STEP 4: Preparing training data...")
    print("=" * 80)
    prepare_training_data(cities=cities, bitre_file=bitre_file)

    # Step 5: Fetch latest month via Flightera API
    print("\n" + "=" * 80)
    print("STEP 5: Fetching latest flight data from Flightera API...")
    print("=" * 80)
    try:
        prev = datetime.now().replace(day=1) - relativedelta(months=1)
        target_ym = prev.strftime('%Y-%m')
        training_path = os.path.join(DATA_PROCESSED, 'ml_training_data_multiroute_hols.csv')
        df_check = pd.read_csv(training_path, usecols=['year_month'])
        if (df_check['year_month'] == target_ym).any():
            print(f"  {target_ym} already present in training data. Skipping Flightera update.")
        else:
            append_flightera_data(target_ym, cities=cities)
    except Exception as e:
        print(f"Flightera update skipped: {e}")

    print("\n" + "=" * 80)
    print("Data update complete.")
    print("=" * 80)


# ===========================================================================
# 8. Model and Data Loading (for Streamlit app)
# ===========================================================================

def load_models(models_dir=None):
    """
    Load all saved model artifacts.

    Returns dict with keys: ridge, rf_reg, logreg, rf_clf, xgb_clf, scaler, nn_reg, nn_clf
    """
    if models_dir is None:
        models_dir = NOWCASTING_MODELS_DIR

    models = {}
    model_files = {
        'ridge': 'ridge_regressor.pkl',
        'rf_reg': 'rf_regressor.pkl',
        'logreg': 'logreg_classifier.pkl',
        'rf_clf': 'rf_classifier.pkl',
        'xgb_clf': 'xgb_classifier.pkl',
        'scaler': 'scaler.pkl',
    }
    for key, filename in model_files.items():
        path = os.path.join(models_dir, filename)
        if os.path.exists(path):
            models[key] = joblib.load(path)
        else:
            print(f"Warning: {path} not found")

    # Load Keras neural network models if available
    try:
        from tensorflow import keras
        nn_reg_path = os.path.join(models_dir, 'nn_regressor.keras')
        nn_clf_path = os.path.join(models_dir, 'nn_classifier.keras')
        if os.path.exists(nn_reg_path):
            models['nn_reg'] = keras.models.load_model(nn_reg_path)
        if os.path.exists(nn_clf_path):
            models['nn_clf'] = keras.models.load_model(nn_clf_path)
    except ImportError:
        pass  # TensorFlow not available

    return models


def load_metadata(models_dir=None):
    """Load metadata.json containing feature names, metrics, etc."""
    if models_dir is None:
        models_dir = NOWCASTING_MODELS_DIR
    path = os.path.join(models_dir, 'metadata.json')
    with open(path, 'r') as f:
        return json.load(f)


def load_training_data(filename='ml_training_data_multiroute_hols.csv'):
    """Load the merged training data CSV."""
    path = os.path.join(DATA_PROCESSED, filename)
    return pd.read_csv(path)


def load_load_factor_data(filepath=None):
    """
    Load market-wide monthly load factor from the BITRE Monthly Airline Performance Excel.

    The load factor is the domestic passenger load factor (pax_load_factor_pct / 100),
    aggregated across all airlines for each calendar month. It is used as an
    additional feature in the forecasting models (lag1 + exponential transform).

    Parameters
    ----------
    filepath : str or None
        Path to the Excel file. If None, searches DATA_RAW for known filenames
        listed in MONTHLY_AIRLINE_PERFORMANCE_CANDIDATES.

    Returns
    -------
    pd.DataFrame
        Columns: ['year_month', 'load_factor'] where load_factor is in [0, 1].
        COVID period (Apr–Dec 2020) is excluded. Rows before 2009 are excluded.

    Raises
    ------
    FileNotFoundError
        If no Monthly Airline Performance file is found.
    """
    if filepath is None:
        for candidate in MONTHLY_AIRLINE_PERFORMANCE_CANDIDATES:
            path = os.path.join(DATA_RAW, candidate)
            if os.path.exists(path):
                filepath = path
                break
        if filepath is None:
            raise FileNotFoundError(
                "No Monthly Airline Performance file found in data/raw/. "
                f"Expected one of: {MONTHLY_AIRLINE_PERFORMANCE_CANDIDATES}"
            )

    df_activity = pd.read_excel(filepath, sheet_name='Domestic airlines', header=None, skiprows=8)
    df_activity.columns = [
        'year', 'month_name', 'hours_flown', 'aircraft_km_flown_000', 'aircraft_departures',
        'total_rev_pax_ud', 'freight_tonnes_ud', 'mail_tonnes_ud',
        'total_rev_pax_tob', 'total_rev_pax_tob_inc_intl',
        'freight_tonnes_tob', 'mail_tonnes_tob',
        'total_rpk_000', 'pax_tonne_km_000', 'freight_tonne_km_000',
        'mail_tonne_km_000', 'total_tonne_km_000',
        'available_seat_km_000', 'available_tonne_km_000', 'available_seats_000',
        'pax_load_factor_pct', 'weight_load_factor_pct',
        'total_charter_pax_tob', 'charter_aircraft_departures',
    ]

    valid_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'June',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    df_activity['year_numeric'] = pd.to_numeric(df_activity['year'], errors='coerce')
    df_act = df_activity[
        df_activity['year_numeric'].notna() &
        df_activity['month_name'].isin(valid_months)
    ].copy()

    df_act['year'] = df_act['year_numeric'].astype(int)
    df_act['month_name'] = df_act['month_name'].replace({'June': 'Jun'})
    month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    df_act['month_num'] = df_act['month_name'].map(month_map)
    df_act['year_month'] = (
        df_act['year'].astype(str) + '-' +
        df_act['month_num'].astype(str).str.zfill(2)
    )
    df_act['load_factor'] = pd.to_numeric(df_act['pax_load_factor_pct'], errors='coerce') / 100.0

    # Exclude COVID anomaly period
    covid_mask = (df_act['year'] == 2020) & (df_act['month_num'] >= 4)
    df_act = df_act[~covid_mask & (df_act['year'] >= 2009)].copy()

    df_lf = df_act[['year_month', 'load_factor']].groupby('year_month', as_index=False).mean()
    return df_lf


def load_forecasting_models(models_dir=None):
    """
    Load forecasting model artifacts from models/forecasting/.

    Returns dict with keys: ridge, rf_reg, logreg, rf_clf, xgb_clf, scaler,
    and optionally nn_reg, nn_clf (Keras neural network models).
    """
    if models_dir is None:
        models_dir = FORECASTING_MODELS_DIR

    models = {}
    model_files = {
        'ridge': 'ridge_regressor.pkl',
        'rf_reg': 'rf_regressor.pkl',
        'logreg': 'logreg_classifier.pkl',
        'rf_clf': 'rf_classifier.pkl',
        'xgb_clf': 'xgb_classifier.pkl',
        'scaler': 'scaler.pkl',
    }
    for key, filename in model_files.items():
        path = os.path.join(models_dir, filename)
        if os.path.exists(path):
            models[key] = joblib.load(path)
        else:
            print(f"Warning: {path} not found")

    # Load Keras neural network models if available
    try:
        from tensorflow import keras
        nn_reg_path = os.path.join(models_dir, 'nn_regressor.keras')
        nn_clf_path = os.path.join(models_dir, 'nn_classifier.keras')
        if os.path.exists(nn_reg_path):
            models['nn_reg'] = keras.models.load_model(nn_reg_path)
        if os.path.exists(nn_clf_path):
            models['nn_clf'] = keras.models.load_model(nn_clf_path)
    except ImportError:
        pass  # TensorFlow not available

    return models


def load_forecasting_metadata(models_dir=None):
    """Load forecasting metadata.json."""
    if models_dir is None:
        models_dir = FORECASTING_MODELS_DIR
    path = os.path.join(models_dir, 'metadata.json')
    with open(path, 'r') as f:
        return json.load(f)


# ===========================================================================
# CLI entry point
# ===========================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='FLAPS data pipeline')
    parser.add_argument('--update', action='store_true', help='Run full data update')
    parser.add_argument('--weather-only', action='store_true', help='Only update weather data')
    parser.add_argument('--cities', nargs='+', default=None,
                        help='Cities to process (full names)')
    args = parser.parse_args()

    if args.weather_only:
        codes = [CITY_MAPPING[c] for c in (args.cities or DEFAULT_CITIES)]
        download_bom_weather_data(codes)
        for code in codes:
            compute_weather_features(code)
    elif args.update:
        update_all_data(args.cities)
    else:
        parser.print_help()
