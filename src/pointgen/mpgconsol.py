import os
import zipfile
import math
import pandas as pd
from datetime import time

##############################################################################
# CONFIG
##############################################################################

SEPTA_BUS_SUBWAY_GTFS_PATH = "./gtfs/SEPTABus"  # or e.g., "/path/to/septa_bus_subway.zip"
SEPTA_REGIONAL_RAIL_GTFS_PATH = "./gtfs/SEPTARail"
PATCO_GTFS_PATH = "./gtfs/PATCO"

# Minimum frequency (in minutes) for bus routes only
BUS_FREQUENCY = 15

# Time window for checking bus frequency
START_TIME = time(6, 0)   # 5:00 AM
END_TIME = time(20, 0)    # 8:00 PM

# Clustering radius in meters
CLUSTER_RADIUS_M = 150

OUTPUT_KML = "./out/final_consolidated_stations.kml"

##############################################################################
# 1) LOAD GTFS
##############################################################################
def load_gtfs(gtfs_path):
    """Load GTFS files from a directory or ZIP into a dict of DataFrames."""
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

##############################################################################
# 2) GET WEEKDAY SERVICE IDS
##############################################################################
def get_weekday_service_ids(calendar_df, calendar_dates_df=None):
    """Return service_ids that operate Monday-Friday (basic approach)."""
    if calendar_df.empty:
        return set()
    
    weekday_services = calendar_df[
        (calendar_df['monday'] == 1) &
        (calendar_df['tuesday'] == 1) &
        (calendar_df['wednesday'] == 1) &
        (calendar_df['thursday'] == 1) &
        (calendar_df['friday'] == 1)
    ]['service_id'].unique().tolist()
    return set(weekday_services)

##############################################################################
# 3) TIME PARSING
##############################################################################
def parse_time_to_minutes(timestr):
    """Parse HH:MM:SS into minutes from midnight. Handles 24+ hour times."""
    if pd.isna(timestr):
        return None
    parts = timestr.split(":")
    if len(parts) < 2:
        return None
    hour = int(parts[0])
    minute = int(parts[1])
    return hour * 60 + minute

def time_obj_to_minutes(t):
    return t.hour * 60 + t.minute

##############################################################################
# 4) FILTER BUS STOPS BY REQUIRED FREQUENCY
##############################################################################
def filter_bus_stops_by_frequency(gtfs_files, max_headway_minutes, start_t, end_t, service_ids):
    """
    For routes where route_type=3 (bus), return stop_ids that meet 'max_headway_minutes'
    throughout [start_t, end_t].
    """
    routes_df = gtfs_files["routes"]
    trips_df = gtfs_files["trips"]
    stop_times_df = gtfs_files["stop_times"]
    
    if routes_df.empty or trips_df.empty or stop_times_df.empty:
        return set()
    
    bus_routes = routes_df[routes_df["route_type"] == 3]["route_id"].unique()
    trips_weekday_bus = trips_df[
        (trips_df["service_id"].isin(service_ids)) &
        (trips_df["route_id"].isin(bus_routes))
    ]
    
    if trips_weekday_bus.empty:
        return set()
    
    merged = stop_times_df.merge(
        trips_weekday_bus[["trip_id", "route_id"]],
        on="trip_id", how="inner"
    )
    merged["arrival_minutes"] = merged["arrival_time"].apply(parse_time_to_minutes)

    start_m = time_obj_to_minutes(start_t)
    end_m = time_obj_to_minutes(end_t)

    # Filter to arrivals within the time window
    merged = merged[
        (merged["arrival_minutes"] >= start_m) &
        (merged["arrival_minutes"] <= end_m)
    ].copy()
    
    qualifying_stops = set()
    for stop_id, group in merged.groupby("stop_id"):
        arrivals = sorted(group["arrival_minutes"].unique())
        if not arrivals:
            continue
        
        # Check gap from start to first arrival
        if arrivals[0] - start_m > max_headway_minutes:
            continue
        
        # Check consecutive gaps
        too_large_gap = False
        for i in range(len(arrivals)-1):
            if arrivals[i+1] - arrivals[i] > max_headway_minutes:
                too_large_gap = True
                break
        if too_large_gap:
            continue
        
        # Check gap from last arrival to end
        if end_m - arrivals[-1] > max_headway_minutes:
            continue
        
        qualifying_stops.add(stop_id)
    
    return qualifying_stops

##############################################################################
# 5) GATHER ROUTES SERVING EACH STOP
##############################################################################
def get_stop_routes(gtfs_files):
    """
    Return a DataFrame with columns: [stop_id, routes],
    where 'routes' is a set of route short_names (or route_ids if short_name is missing).
    """
    routes_df = gtfs_files["routes"]
    trips_df = gtfs_files["trips"]
    stop_times_df = gtfs_files["stop_times"]
    
    if routes_df.empty or trips_df.empty or stop_times_df.empty:
        return pd.DataFrame(columns=["stop_id", "routes"])
    
    # Merge trips->routes to get route_short_name or fallback to route_id
    merged_trips = trips_df.merge(routes_df, on="route_id", how="left", suffixes=("_trip","_route"))
    # Now merge stop_times->merged_trips
    st_rt = stop_times_df.merge(merged_trips, on="trip_id", how="left")
    
    # We'll define "route_name" as route_short_name if present, else route_id
    st_rt["route_name"] = st_rt["route_short_name"].fillna(st_rt["route_id"])
    
    # Group by stop_id, collecting route_names into a set
    grouped = st_rt.groupby("stop_id")["route_name"].apply(lambda x: set(x.dropna()))
    # Convert to a DataFrame
    stop_routes_df = grouped.reset_index().rename(columns={"route_name":"routes"})
    return stop_routes_df

##############################################################################
# 6) HAVERSINE FOR DISTANCE
##############################################################################
def haversine_m(lat1, lon1, lat2, lon2):
    """
    Returns distance in meters between lat/lon coords using Haversine formula.
    """
    R = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi/2)**2
        + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

##############################################################################
# 7) CONSOLIDATE (CLUSTER) STOPS WITHIN RADIUS
##############################################################################
def consolidate_stops_within_radius(stops_df, radius_m=100):
    """
    Clusters stops within 'radius_m' meters. 
    stops_df should have columns:
      - stop_name
      - stop_lat
      - stop_lon
      - routes (a set of route names)
    We produce a new DF with columns:
      - consolidated_name
      - lat
      - lon
      - route_list
    Where consolidated_name = "StopA / StopB (Route1, Route2, ...)" 
      if multiple stops are combined.
    """
    clusters = []
    for _, row in stops_df.iterrows():
        s_name = row["stop_name"]
        s_lat  = row["stop_lat"]
        s_lon  = row["stop_lon"]
        s_routes = row["routes"] if "routes" in row else set()
        
        found_cluster = False
        for cl in clusters:
            dist = haversine_m(s_lat, s_lon, cl["lat"], cl["lon"])
            if dist <= radius_m:
                # Merge
                cl["names"].append(s_name)
                cl["routes"].update(s_routes)
                cl["all_lats"].append(s_lat)
                cl["all_lons"].append(s_lon)
                # Recompute centroid
                cl["lat"] = sum(cl["all_lats"]) / len(cl["all_lats"])
                cl["lon"] = sum(cl["all_lons"]) / len(cl["all_lons"])
                found_cluster = True
                break
        
        if not found_cluster:
            clusters.append({
                "names": [s_name],
                "routes": set(s_routes),
                "lat": s_lat,
                "lon": s_lon,
                "all_lats": [s_lat],
                "all_lons": [s_lon]
            })
    
    # Build final DataFrame
    records = []
    for cl in clusters:
        # Unique stop names
        all_names = sorted(set(cl["names"]))
        # Join with slash
        combined_name = " / ".join(all_names)
        # Combine routes
        route_list = sorted(cl["routes"])
        if route_list:
            combined_name += f" ({', '.join(route_list)})"
        
        records.append({
            "consolidated_name": combined_name,
            "lat": cl["lat"],
            "lon": cl["lon"],
            "route_list": route_list
        })
    
    return pd.DataFrame(records)

##############################################################################
# 8) GENERATE KML
##############################################################################
def generate_kml(consolidated_df, output_file):
    """
    Exports KML with <name> set to 'consolidated_name', and coordinates are (lon, lat).
    Replaces & with &amp; to avoid parsing issues.
    """
    kml_header = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
"""
    kml_footer = """</Document>
</kml>
"""
    placemarks = []
    for _, row in consolidated_df.iterrows():
        name = str(row["consolidated_name"]).replace("&", "&amp;")
        lat = row["lat"]
        lon = row["lon"]
        pm = f"""
  <Placemark>
    <name>{name}</name>
    <Point>
      <coordinates>{lon},{lat},0</coordinates>
    </Point>
  </Placemark>
"""
        placemarks.append(pm)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(kml_header)
        for pm in placemarks:
            f.write(pm)
        f.write(kml_footer)

##############################################################################
# MAIN
##############################################################################
def main():
    #################
    # A) Load GTFS
    #################
    gtfs_bs = load_gtfs(SEPTA_BUS_SUBWAY_GTFS_PATH)
    gtfs_rr = load_gtfs(SEPTA_REGIONAL_RAIL_GTFS_PATH)
    gtfs_patco = load_gtfs(PATCO_GTFS_PATH)

    # B) Weekday service IDs
    bs_sids = get_weekday_service_ids(gtfs_bs["calendar"], gtfs_bs["calendar_dates"])
    rr_sids = get_weekday_service_ids(gtfs_rr["calendar"], gtfs_rr["calendar_dates"])
    patco_sids = get_weekday_service_ids(gtfs_patco["calendar"], gtfs_patco["calendar_dates"])

    # C) Filter Bus stops by frequency
    bus_stops_qualified = filter_bus_stops_by_frequency(
        gtfs_bs, BUS_FREQUENCY, START_TIME, END_TIME, bs_sids
    )

    # D) Non-bus stops in Bus/Subway feed => all stops for routes with route_type != 3
    routes_bs = gtfs_bs["routes"]
    trips_bs = gtfs_bs["trips"]
    stop_times_bs = gtfs_bs["stop_times"]

    non_bus_stops = set()
    if not routes_bs.empty and not trips_bs.empty and not stop_times_bs.empty:
        nb_routes = routes_bs[routes_bs["route_type"] != 3]["route_id"].unique()
        trips_non_bus = trips_bs[trips_bs["route_id"].isin(nb_routes)]
        merged_nb = stop_times_bs.merge(
            trips_non_bus[["trip_id","route_id"]], on="trip_id", how="inner"
        )
        non_bus_stops = set(merged_nb["stop_id"].unique())

    # E) Consolidate Bus & Subway stops => bus freq pass OR non-bus
    bs_all_stops = bus_stops_qualified.union(non_bus_stops)

    # F) Regional Rail => all stops, ignoring frequency
    rr_stops_df = gtfs_rr["stops"]
    rr_all_stops = set(rr_stops_df["stop_id"].unique())

    # G) PATCO => all stops, ignoring frequency
    patco_stops_df = gtfs_patco["stops"]
    patco_all_stops = set(patco_stops_df["stop_id"].unique())

    #################
    # H) Build full stops data for each feed & gather routes
    #################

    # 1. For the Bus/Subway feed
    bs_stops_df = gtfs_bs["stops"]
    if not bs_stops_df.empty:
        # Filter to stops we want
        bs_filtered = bs_stops_df[bs_stops_df["stop_id"].isin(bs_all_stops)].copy()
        # Gather routes sets
        bs_stop_routes_df = get_stop_routes(gtfs_bs)  # [stop_id, routes]
        bs_filtered = bs_filtered.merge(bs_stop_routes_df, on="stop_id", how="left")
        bs_filtered["routes"] = bs_filtered["routes"].apply(lambda x: x if isinstance(x, set) else set())
        # Tag them
        bs_filtered["agency"] = "SEPTA_BusSubway"
    else:
        bs_filtered = pd.DataFrame(columns=["stop_id","stop_name","stop_lat","stop_lon","routes","agency"])

    # 2. Regional Rail
    if not rr_stops_df.empty:
        rr_stop_routes_df = get_stop_routes(gtfs_rr)  # [stop_id, routes]
        rr_filtered = rr_stops_df[rr_stops_df["stop_id"].isin(rr_all_stops)].copy()
        rr_filtered = rr_filtered.merge(rr_stop_routes_df, on="stop_id", how="left")
        rr_filtered["routes"] = rr_filtered["routes"].apply(lambda x: x if isinstance(x, set) else set())
        rr_filtered["agency"] = "SEPTA_RegionalRail"
    else:
        rr_filtered = pd.DataFrame(columns=["stop_id","stop_name","stop_lat","stop_lon","routes","agency"])

    # 3. PATCO
    if not patco_stops_df.empty:
        patco_stop_routes_df = get_stop_routes(gtfs_patco)  # [stop_id, routes]
        patco_filtered = patco_stops_df[patco_stops_df["stop_id"].isin(patco_all_stops)].copy()
        patco_filtered = patco_filtered.merge(patco_stop_routes_df, on="stop_id", how="left")
        patco_filtered["routes"] = patco_filtered["routes"].apply(lambda x: x if isinstance(x, set) else set())
        patco_filtered["agency"] = "PATCO"
    else:
        patco_filtered = pd.DataFrame(columns=["stop_id","stop_name","stop_lat","stop_lon","routes","agency"])

    #################
    # I) Combine
    #################
    all_stops = pd.concat([bs_filtered, rr_filtered, patco_filtered], ignore_index=True)

    # The key columns we'll use for consolidation:
    #   stop_name, stop_lat, stop_lon, routes
    # If any are missing, fill them
    for col in ["stop_name","stop_lat","stop_lon","routes"]:
        if col not in all_stops.columns:
            all_stops[col] = None

    # Just rename them for clarity
    df_for_cluster = all_stops[["stop_name","stop_lat","stop_lon","routes"]].copy()

    #################
    # J) Cluster / Consolidate
    #################
    consolidated_df = consolidate_stops_within_radius(df_for_cluster, radius_m=CLUSTER_RADIUS_M)

    #################
    # K) Generate KML
    #################
    generate_kml(consolidated_df, OUTPUT_KML)
    print(f"Done! KML with consolidated stations saved to: {OUTPUT_KML}")

##############################################################################
# RUN
##############################################################################
if __name__ == "__main__":
    main()
