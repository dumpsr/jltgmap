import pandas as pd
import os
from datetime import time
import zipfile

##############################
# User Configurable Variables
##############################

# Paths to the extracted GTFS directories or ZIP files
SEPTA_BUS_SUBWAY_GTFS_PATH = "./gtfs/SEPTABus"  # or e.g., "/path/to/septa_bus_subway.zip"
SEPTA_REGIONAL_RAIL_GTFS_PATH = "./gtfs/SEPTARail"
PATCO_GTFS_PATH = "./gtfs/PATCO"

# The service frequency requirements
BUS_SUBWAY_PATCO_FREQUENCY = 15 # minutes
REGIONAL_RAIL_FREQUENCY = 30 # minutes

# Weekday filter (e.g., Monday=1, Tuesday=1, etc. in calendar.txt)
WEEKDAY_SERVICE_IDS = None  # We'll compute this from calendar.txt or calendar_dates.txt

# The time window for analysis (e.g., 5:00 AM to 8:00 PM)
START_TIME = time(6, 0)   # 5:00 AM
END_TIME = time(20, 0)    # 8:00 PM

# Output KML file
OUTPUT_KML = "./out/stations_frequent_service.kml"


##############################
# Helper Functions
##############################

def load_gtfs(gtfs_path):
    """
    Load GTFS files (agency, stops, routes, trips, stop_times, calendar, etc.)
    from a directory or a .zip file.
    Returns a dict of DataFrames.
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
    files["agency"] = read_csv("agency.txt")
    files["stops"] = read_csv("stops.txt")
    files["routes"] = read_csv("routes.txt")
    files["trips"] = read_csv("trips.txt")
    files["stop_times"] = read_csv("stop_times.txt")
    files["calendar"] = read_csv("calendar.txt")
    files["calendar_dates"] = read_csv("calendar_dates.txt")
    return files


def get_weekday_service_ids(calendar_df, calendar_dates_df=None):
    """
    For simplicity, gather all service_ids that operate Monday–Friday.
    This function can be adapted/expanded based on actual date ranges
    or your desired set of weekday(s).
    """
    # Filter to entries that operate on weekdays (Monday=1, Tuesday=1, etc.)
    # and are active (start_date <= today <= end_date) if you want to be date-specific
    weekday_services = calendar_df[
        (calendar_df['monday'] == 1) &
        (calendar_df['tuesday'] == 1) &
        (calendar_df['wednesday'] == 1) &
        (calendar_df['thursday'] == 1) &
        (calendar_df['friday'] == 1)
    ]['service_id'].unique().tolist()
    
    # If you want to incorporate calendar_dates, handle exceptions here:
    # e.g., remove service_ids not operating on certain weekdays,
    # or add service_ids that do. For simplicity, we’ll skip advanced logic.
    
    return set(weekday_services)


def parse_time_to_minutes(timestr):
    """
    Parse HH:MM:SS (or HH:MM) to minutes from midnight.
    If times are above 24 hours, GTFS can do that (e.g., 25:30:00),
    so this function accounts for it.
    """
    parts = timestr.split(":")
    if len(parts) < 2:
        return None
    
    hour = int(parts[0])
    minute = int(parts[1])
    return hour * 60 + minute


def time_obj_to_minutes(tobj):
    """Convert a Python time object to minutes from midnight."""
    return tobj.hour * 60 + tobj.minute


def filter_stops_by_frequency(
    gtfs_files, 
    max_headway_minutes, 
    start_time, 
    end_time, 
    service_ids
):
    """
    Given GTFS DataFrames and a max headway requirement (in minutes),
    return a set of stop_ids that have at least that frequency
    throughout the specified time window on a typical weekday.
    """
    stops_df = gtfs_files["stops"]
    trips_df = gtfs_files["trips"]
    stop_times_df = gtfs_files["stop_times"]

    # Filter to only weekday service
    trips_weekday = trips_df[trips_df["service_id"].isin(service_ids)]
    
    # Merge trips with stop_times for arrival/departure times
    merged = stop_times_df.merge(
        trips_weekday[["trip_id", "route_id"]], 
        on="trip_id", 
        how="inner"
    )

    # Convert time strings to minutes from midnight
    merged["arrival_minutes"] = merged["arrival_time"].apply(parse_time_to_minutes)
    merged["departure_minutes"] = merged["departure_time"].apply(parse_time_to_minutes)

    # Filter to the desired time window
    start_m = time_obj_to_minutes(start_time)
    end_m = time_obj_to_minutes(end_time)
    
    merged = merged[
        (merged["arrival_minutes"] >= start_m) & 
        (merged["arrival_minutes"] <= end_m)
    ].copy()

    # We want to see how often a stop is served. A straightforward approach:
    # 1) Group by stop_id
    # 2) For each stop, look at the sorted arrival times
    # 3) Check the gaps between consecutive trips within that window
    #    If all consecutive arrivals are within 'max_headway_minutes', 
    #    then the stop meets the requirement.
    # 
    # A simpler approximation is to check that in every hour block (or 15/30-min block),
    # there's at least one trip. But we’ll do a direct check of actual consecutive trips.
    
    qualifying_stops = set()
    
    for stop_id, group in merged.groupby("stop_id"):
        arrival_times = sorted(group["arrival_minutes"].unique())
        
        # If no arrivals, skip
        if not arrival_times:
            continue
        
        # Check consecutive gaps throughout the window
        # Also consider the gap from start of the window to the first trip,
        # and the last trip to the end of the window.
        
        # Gap from start of window to first trip:
        if arrival_times[0] - start_m > max_headway_minutes:
            continue
        
        # Gaps between consecutive trips:
        too_large_gap_found = False
        for i in range(len(arrival_times) - 1):
            if (arrival_times[i+1] - arrival_times[i]) > max_headway_minutes:
                too_large_gap_found = True
                break
        
        if too_large_gap_found:
            continue
        
        # Gap from last trip to end of window
        if end_m - arrival_times[-1] > max_headway_minutes:
            continue
        
        # If we reach here, all gaps are within threshold
        qualifying_stops.add(stop_id)
    
    return qualifying_stops


def generate_kml(stops_df, stop_ids, kml_file):
    """
    Generate a KML file listing all stops in stop_ids with placemarks.
    The stops_df should contain 'stop_id', 'stop_name', 'stop_lat', 'stop_lon'.
    """
    kml_header = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
"""
    kml_footer = """</Document>
</kml>
"""
    placemarks = []
    for _, row in stops_df[stops_df["stop_id"].isin(stop_ids)].iterrows():
        # Replace & with &amp;
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
# Main Logic
##############################

def main():
    # 1. Load each GTFS feed
    bus_subway_gtfs = load_gtfs(SEPTA_BUS_SUBWAY_GTFS_PATH)
    regional_rail_gtfs = load_gtfs(SEPTA_REGIONAL_RAIL_GTFS_PATH)
    patco_gtfs = load_gtfs(PATCO_GTFS_PATH)

    # 2. Get weekday service IDs for each feed
    bus_subway_service_ids = get_weekday_service_ids(
        bus_subway_gtfs["calendar"], bus_subway_gtfs["calendar_dates"]
    )
    regional_rail_service_ids = get_weekday_service_ids(
        regional_rail_gtfs["calendar"], regional_rail_gtfs["calendar_dates"]
    )
    patco_service_ids = get_weekday_service_ids(
        patco_gtfs["calendar"], patco_gtfs["calendar_dates"]
    )

    # 3. Filter stops by frequency
    #    - bus_subway stops: every 15 min or better
    bus_subway_stops_15 = filter_stops_by_frequency(
        bus_subway_gtfs, 
        BUS_SUBWAY_PATCO_FREQUENCY, 
        START_TIME, 
        END_TIME, 
        bus_subway_service_ids
    )

    #    - regional rail: every 30 min or better
    regional_rail_stops_30 = filter_stops_by_frequency(
        regional_rail_gtfs,
        REGIONAL_RAIL_FREQUENCY,
        START_TIME,
        END_TIME,
        regional_rail_service_ids
    )

    #    - PATCO stops: every 15 min or better
    patco_stops_15 = filter_stops_by_frequency(
        patco_gtfs,
        BUS_SUBWAY_PATCO_FREQUENCY,
        START_TIME,
        END_TIME,
        patco_service_ids
    )

    # 4. Combine them
    #    Because these are different GTFS sets, each will have its own stops table.
    #    We’ll simply create separate sets of KML placemarks (or combine them with
    #    unique IDs). Below we’ll merge them into one file for convenience.

    # Build a big list (stop_id, lat, lon, name) for each feed
    # (We need to keep them separate because a "stop_id" in one feed might clash with another.)
    combined_stops = []
    
    # For bus_subway
    stops_df_bs = bus_subway_gtfs["stops"]
    stops_bs_qualified = stops_df_bs[stops_df_bs["stop_id"].isin(bus_subway_stops_15)]
    stops_bs_qualified["source_feed"] = "SEPTA_Bus_Subway"
    combined_stops.append(stops_bs_qualified)

    # For regional rail
    stops_df_rr = regional_rail_gtfs["stops"]
    stops_rr_qualified = stops_df_rr[stops_df_rr["stop_id"].isin(regional_rail_stops_30)]
    stops_rr_qualified["source_feed"] = "SEPTA_Regional_Rail"
    combined_stops.append(stops_rr_qualified)

    # For PATCO
    stops_df_patco = patco_gtfs["stops"]
    stops_patco_qualified = stops_df_patco[stops_df_patco["stop_id"].isin(patco_stops_15)]
    stops_patco_qualified["source_feed"] = "PATCO"
    combined_stops.append(stops_patco_qualified)

    all_qualified_stops = pd.concat(combined_stops, ignore_index=True)

    # 5. Write to KML
    # If you’d like them all in one file, just pass the combined DataFrame’s IDs to generate_kml.
    generate_kml(
        all_qualified_stops,
        all_qualified_stops["stop_id"],
        OUTPUT_KML
    )

    print(f"KML with frequent-service stations saved to {OUTPUT_KML}")


if __name__ == "__main__":
    main()
