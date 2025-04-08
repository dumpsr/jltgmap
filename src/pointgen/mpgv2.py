import pandas as pd
import os
from datetime import time
import zipfile

##############################
# User Configurable Variables
##############################

# Paths to GTFS (directory or ZIP). Update these to real paths:
SEPTA_BUS_SUBWAY_GTFS_PATH = "./gtfs/SEPTABus"  # or e.g., "/path/to/septa_bus_subway.zip"
SEPTA_REGIONAL_RAIL_GTFS_PATH = "./gtfs/SEPTARail"
PATCO_GTFS_PATH = "./gtfs/PATCO"

# Minimum frequency (in minutes) for bus routes only
BUS_FREQUENCY = 15

# Time window for checking bus frequency (e.g., 5 AM to 8 PM)
START_TIME = time(6, 0)   # 5:00 AM
END_TIME = time(20, 0)    # 8:00 PM

# Output KML file
OUTPUT_KML = "./out/stations_including_nonbus.kml"

##############################
# Helper Functions
##############################

def load_gtfs(gtfs_path):
    """
    Load GTFS files (agency, stops, routes, trips, stop_times, calendar, calendar_dates)
    from a directory or a .zip file. Returns a dict of DataFrames.
    """
    def read_csv_from_dir(directory, filename):
        file_path = os.path.join(directory, filename)
        if not os.path.exists(file_path):
            return pd.DataFrame()
        return pd.read_csv(file_path)
    
    def read_csv_from_zip(zip_path, filename):
        with zipfile.ZipFile(zip_path, 'r') as z:
            if filename not in z.namelist():
                return pd.DataFrame()
            with z.open(filename) as f:
                return pd.read_csv(f)
    
    # Decide whether we're reading from a directory or a zip
    if os.path.isdir(gtfs_path):
        read_csv = lambda fname: read_csv_from_dir(gtfs_path, fname)
    else:
        read_csv = lambda fname: read_csv_from_zip(gtfs_path, fname)
    
    files = {}
    for fname in [
        "agency.txt",
        "stops.txt",
        "routes.txt",
        "trips.txt",
        "stop_times.txt",
        "calendar.txt",
        "calendar_dates.txt"
    ]:
        files[fname.replace(".txt","")] = read_csv(fname)
    
    return files

def get_weekday_service_ids(calendar_df, calendar_dates_df=None):
    """
    Returns service_ids that operate Monday-Friday. Adjust as needed if you want
    more detailed date-range filtering.
    """
    if calendar_df.empty:
        return set()
    
    # Basic filter for Monday-Friday = 1
    weekday_services = calendar_df[
        (calendar_df['monday'] == 1) &
        (calendar_df['tuesday'] == 1) &
        (calendar_df['wednesday'] == 1) &
        (calendar_df['thursday'] == 1) &
        (calendar_df['friday'] == 1)
    ]['service_id'].unique().tolist()
    
    # If you need to account for calendar_dates, do so here; we skip advanced logic.
    return set(weekday_services)

def parse_time_to_minutes(timestr):
    """Parses HH:MM:SS (or HH:MM) into minutes from midnight. Handles 24+ hour times."""
    if pd.isna(timestr):
        return None
    parts = timestr.split(":")
    if len(parts) < 2:
        return None
    hour = int(parts[0])
    minute = int(parts[1])
    return hour * 60 + minute

def time_obj_to_minutes(tobj):
    """Convert a Python time object to minutes from midnight."""
    return tobj.hour * 60 + tobj.minute

def filter_bus_stops_by_frequency(gtfs_files, max_headway_minutes, start_time, end_time, service_ids):
    """
    Only used for bus routes (route_type=3).
    Returns a set of stop_ids that have at least that frequency
    throughout the specified time window on a typical weekday.
    """
    if any(df.empty for df in [gtfs_files["routes"], gtfs_files["trips"], gtfs_files["stop_times"]]):
        return set()
    
    routes_df = gtfs_files["routes"]
    trips_df = gtfs_files["trips"]
    stop_times_df = gtfs_files["stop_times"]
    
    # Identify bus routes
    bus_routes = routes_df[routes_df["route_type"] == 3]["route_id"].unique()
    # Limit trips to those with weekday service AND route in bus_routes
    trips_weekday_bus = trips_df[
        (trips_df["service_id"].isin(service_ids)) &
        (trips_df["route_id"].isin(bus_routes))
    ]
    
    if trips_weekday_bus.empty:
        return set()
    
    # Merge with stop_times
    merged = stop_times_df.merge(
        trips_weekday_bus[["trip_id", "route_id"]],
        on="trip_id",
        how="inner"
    )
    # Convert arrival times to minutes
    merged["arrival_minutes"] = merged["arrival_time"].apply(parse_time_to_minutes)

    start_m = time_obj_to_minutes(start_time)
    end_m = time_obj_to_minutes(end_time)

    # Keep only arrivals in the time window
    merged = merged[
        (merged["arrival_minutes"] >= start_m) &
        (merged["arrival_minutes"] <= end_m)
    ].copy()
    
    # Group by stop_id and check consecutive arrival gaps
    qualifying_stops = set()
    
    for stop_id, group in merged.groupby("stop_id"):
        arrivals = sorted(group["arrival_minutes"].unique())
        if not arrivals:
            continue
        
        # Check gap from window start to first arrival
        if arrivals[0] - start_m > max_headway_minutes:
            continue
        
        # Check consecutive gaps
        too_large_gap = False
        for i in range(len(arrivals) - 1):
            if arrivals[i+1] - arrivals[i] > max_headway_minutes:
                too_large_gap = True
                break
        if too_large_gap:
            continue
        
        # Check gap from last arrival to window end
        if end_m - arrivals[-1] > max_headway_minutes:
            continue
        
        qualifying_stops.add(stop_id)
    
    return qualifying_stops

def generate_kml(stops_df, stop_ids, kml_file):
    """
    Generate a KML file listing all stops in stop_ids with placemarks.
    The stops_df should contain: stop_id, stop_name, stop_lat, stop_lon.
    """
    kml_header = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
"""
    kml_footer = """</Document>
</kml>
"""
    placemarks = []
    subset = stops_df[stops_df["stop_id"].isin(stop_ids)]
    for _, row in subset.iterrows():
        # Replace & with &amp; in the stop name
        name = str(row["stop_name"]).replace("&", "&amp;")
        lat = row["stop_lat"]
        lon = row["stop_lon"]
        
        placemark = f"""
  <Placemark>
    <name>{name}</name>
    <Point>
      <coordinates>{lon},{lat},0</coordinates>
    </Point>
  </Placemark>
"""
        placemarks.append(placemark)
    
    with open(kml_file, "w", encoding="utf-8") as f:
        f.write(kml_header)
        for pm in placemarks:
            f.write(pm)
        f.write(kml_footer)

##############################
# Main
##############################

def main():
    # 1. Load GTFS
    gtfs_bus_subway = load_gtfs(SEPTA_BUS_SUBWAY_GTFS_PATH)
    gtfs_regional_rail = load_gtfs(SEPTA_REGIONAL_RAIL_GTFS_PATH)
    gtfs_patco = load_gtfs(PATCO_GTFS_PATH)

    # 2. Identify weekday service_ids for each
    bus_subway_service_ids = get_weekday_service_ids(
        gtfs_bus_subway["calendar"], gtfs_bus_subway["calendar_dates"]
    )
    # Regional Rail & PATCO service_ids not actually needed for freq checks, but let's still read them:
    rr_service_ids = get_weekday_service_ids(
        gtfs_regional_rail["calendar"], gtfs_regional_rail["calendar_dates"]
    )
    patco_service_ids = get_weekday_service_ids(
        gtfs_patco["calendar"], gtfs_patco["calendar_dates"]
    )

    # 3. For SEPTA Bus & Subway feed:
    #    (a) Get the stops that are served by bus routes and meet the 15-min frequency requirement
    bus_stops_qualified = filter_bus_stops_by_frequency(
        gtfs_bus_subway,
        max_headway_minutes=BUS_FREQUENCY,
        start_time=START_TIME,
        end_time=END_TIME,
        service_ids=bus_subway_service_ids
    )
    #    (b) Include ALL stops that are served by non-bus routes (i.e., not route_type=3).
    #        This requires finding any stop that’s served by any route_type != 3.
    #        If any route serving a stop is non-bus, we include it automatically.

    routes_bs = gtfs_bus_subway["routes"]
    trips_bs = gtfs_bus_subway["trips"]
    stop_times_bs = gtfs_bus_subway["stop_times"]

    if not routes_bs.empty and not trips_bs.empty and not stop_times_bs.empty:
        # Identify non-bus routes
        non_bus_routes = routes_bs[routes_bs["route_type"] != 3]["route_id"].unique()
        # Trips for non-bus routes
        trips_non_bus = trips_bs[trips_bs["route_id"].isin(non_bus_routes)]
        # Merge to find the stops
        merged_non_bus = stop_times_bs.merge(
            trips_non_bus[["trip_id", "route_id"]],
            on="trip_id",
            how="inner"
        )
        non_bus_stops = set(merged_non_bus["stop_id"].unique())
    else:
        non_bus_stops = set()

    # Combined stops (either bus passes the freq filter, or served by non-bus)
    stops_bus_subway_final = bus_stops_qualified.union(non_bus_stops)

    # 4. For Regional Rail (non-bus) => include ALL stops, ignoring frequency
    rr_stops_df = gtfs_regional_rail["stops"]
    rr_stops_all = set(rr_stops_df["stop_id"].unique())

    # 5. For PATCO (non-bus) => include ALL stops, ignoring frequency
    patco_stops_df = gtfs_patco["stops"]
    patco_stops_all = set(patco_stops_df["stop_id"].unique())

    # 6. Build a combined DataFrame of all relevant stops from each feed
    #    We'll label them to keep them separate (so IDs don't clash).
    
    # ~~~ SEPTA Bus/Subway ~~~
    bs_stops_df = gtfs_bus_subway["stops"]
    bs_use = bs_stops_df[bs_stops_df["stop_id"].isin(stops_bus_subway_final)].copy()
    # Tag them
    bs_use["agency"] = "SEPTA_BusSubway"

    # ~~~ Regional Rail ~~~
    rr_use = rr_stops_df[rr_stops_df["stop_id"].isin(rr_stops_all)].copy()
    rr_use["agency"] = "SEPTA_RegionalRail"

    # ~~~ PATCO ~~~
    patco_use = patco_stops_df[patco_stops_df["stop_id"].isin(patco_stops_all)].copy()
    patco_use["agency"] = "PATCO"

    all_stops = pd.concat([bs_use, rr_use, patco_use], ignore_index=True)

    # 7. Generate KML
    generate_kml(all_stops, all_stops["stop_id"], OUTPUT_KML)

    print(f"KML generated: {OUTPUT_KML}")

if __name__ == "__main__":
    main()
