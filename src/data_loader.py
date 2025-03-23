import json
import fastf1
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Enable FastF1 caching to store previously downloaded data locally.
fastf1.Cache.enable_cache('cache')


# Define constants for the years, session types, data directory, and maximum number of threads.
YEARS = [2024]
SESSION_TYPES = ['Q', 'R']  # Q = Qualifying, R = Race
DATA_DIR = "data/raw"
MAX_CORES = 10  # Maximum number of parallel threads


# =============================================================================
# Utility Functions
# =============================================================================
def get_event_schedule(year):
    # Fetch the race schedule for the given year using FastF1.
    try:
        schedule = fastf1.get_event_schedule(year)
        return schedule
    except Exception as e:
        print(f"Failed to fetch schedule for {year}: {e}")
        return pd.DataFrame()

def save_json(data, filepath):
    # Save a dictionary or list to a JSON file.
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"Saved {filepath}")

def save_weather_json(session, filepath):
    # Extract and save average weather data for a race session.
    try:
        weather = session.weather_data
        # Calculate average values for numeric weather parameters.
        weather_avg = weather.mean(numeric_only=True)
        weather_dict = {
            'AirTemp': weather_avg.get('AirTemperature', None),
            'TrackTemp': weather_avg.get('TrackTemperature', None),
            'Humidity': weather_avg.get('Humidity', None),
            'Pressure': weather_avg.get('Pressure', None),
            'WindSpeed': weather_avg.get('WindSpeed', None)
        }
        # Save the weather data to JSON.
        save_json(weather_dict, filepath)
    except Exception as e:
        print(f"Failed to save weather for {filepath}: {e}")

def load_and_save_session(year, event_name, session_type):
    # Load session data and save results and weather data (if applicable).
    event_id = event_name.replace(" ", "_")
    base_path = f"{DATA_DIR}/{year}_{event_id}_{session_type}"

    try:
        # Load the session data using FastF1.
        session = fastf1.get_session(year, event_name, session_type)
        session.load()
    except Exception as e:
        print(f"Failed to load: {year} {event_name} {session_type} — {e}")
        return

    try:
        # Save session results to JSON file if available.
        if hasattr(session, 'results') and session.results is not None:
            save_json(session.results.to_dict(orient='records'), f"{base_path}_results.json")
    except Exception as e:
        print(f"Failed to save results for {event_name} {session_type}: {e}")

    # Save weather data only for race sessions.
    if session_type == 'R':
        weather_path = f"{base_path}_weather.json"
        save_weather_json(session, weather_path)


# =============================================================================
# Pipeline Runner (Parallel Execution)
# =============================================================================
def run_pipeline_parallel():
    # Create the data directory if it doesn't exist.
    os.makedirs(DATA_DIR, exist_ok=True)
    tasks = []

    # Use ThreadPoolExecutor to parallelize session downloads.
    with ThreadPoolExecutor(max_workers = MAX_CORES) as executor:
        for year in YEARS:
            schedule = get_event_schedule(year)
            if schedule.empty:
                continue

            for _, event in schedule.iterrows():
                event_name = event['EventName']

                # Skip non-race related events like testing, media days, etc.
                if "Test" in event_name or "Pre-Season" in event_name or "Media" in event_name:
                    print(f"Skipping non-race event: {event_name} ({year})")
                    continue

                # Submit a task for each session type (Qualifying and Race).
                for session_type in SESSION_TYPES:
                    tasks.append(executor.submit(load_and_save_session, year, event_name, session_type))

        # Collect results from all completed tasks.
        for future in as_completed(tasks):
            try:
                future.result()
            except Exception as e:
                print(f"Unexpected thread failure: {e}")

run_pipeline_parallel()
