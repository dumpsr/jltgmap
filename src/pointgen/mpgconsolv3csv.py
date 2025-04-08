import os
import zipfile
import math
import pandas as pd
from datetime import time
# from shapely.geometry import Point, Polygon, MultiPolygon
# from shapely.ops import unary_union, transform

##############################################################################
# CONFIG
##############################################################################

SEPTA_BUS_SUBWAY_GTFS_PATH = "./gtfs/SEPTABus"  # or e.g., "/path/to/septa_bus_subway.zip"
SEPTA_REGIONAL_RAIL_GTFS_PATH = "./gtfs/SEPTARail"
PATCO_GTFS_PATH = "./gtfs/PATCO"

BUS_FREQUENCY = 15
START_TIME = time(6, 0)
END_TIME = time(20, 0)

BUS_TOO_CLOSE_RADIUS_M = 250
BUS_MULTI_LINE_RADIUS_M = 100
FINAL_CONSOLIDATION_RADIUS_M = 400

OUTPUT_KML = "./out/final_consolidated_stations.kml"
OUTPUT_CSV = "./out/stations.csv"  # for the ring creation script

##############################################################################
# HELPER FUNCTIONS
##############################################################################

def load_gtfs(gtfs_path):
    """Load GTFS files from a directory or ZIP into a dict of DataFrames."""
    import pandas as pd

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

def parse_time_to_minutes(timestr):
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

def filter_bus_stops_by_frequency(gtfs_files, max_headway_minutes, start_t, end_t, service_ids):
    """For routes where route_type=3 (bus), return stop_ids that meet 'max_headway_minutes' throughout [start_t, end_t]."""
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
    end_m   = time_obj_to_minutes(end_t)

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
        for i in range(len(arrivals) - 1):
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

def get_stop_routes(gtfs_files):
    """Return DataFrame [stop_id, routes], where 'routes' is a set of route names for that stop."""
    routes_df = gtfs_files["routes"]
    trips_df = gtfs_files["trips"]
    stop_times_df = gtfs_files["stop_times"]
    
    if routes_df.empty or trips_df.empty or stop_times_df.empty:
        return pd.DataFrame(columns=["stop_id", "routes"])
    
    merged_trips = trips_df.merge(routes_df, on="route_id", how="left", suffixes=("_trip","_route"))
    st_rt = stop_times_df.merge(merged_trips, on="trip_id", how="left")
    
    st_rt["route_name"] = st_rt["route_short_name"].fillna(st_rt["route_id"])
    
    grouped = st_rt.groupby("stop_id")["route_name"].apply(lambda x: set(x.dropna()))
    stop_routes_df = grouped.reset_index().rename(columns={"route_name":"routes"})
    return stop_routes_df

def haversine_m(lat1, lon1, lat2, lon2):
    """Returns distance in meters between lat/lon coords (WGS84) using Haversine formula."""
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

def filter_close_bus_stops(bus_stops_df, dist_remove=250, dist_multiline=100):
    """
    Among bus stops, remove any stop that is within 'dist_remove' meters of an already-kept stop,
    UNLESS it is within 'dist_multiline' meters and together they serve multiple lines.
    """
    if bus_stops_df.empty:
        return bus_stops_df

    kept_records = []

    for _, row in bus_stops_df.iterrows():
        s_lat = row["stop_lat"]
        s_lon = row["stop_lon"]
        s_routes = row["routes"] if isinstance(row["routes"], set) else set()

        should_keep = True
        for kept in kept_records:
            k_lat = kept["stop_lat"]
            k_lon = kept["stop_lon"]
            k_routes = kept["routes"]

            dist = haversine_m(s_lat, s_lon, k_lat, k_lon)
            if dist < dist_remove:
                union_routes = s_routes.union(k_routes)
                if dist < dist_multiline and len(union_routes) > 1:
                    # keep it
                    should_keep = True
                    break
                else:
                    # discard
                    should_keep = False
                    break
        
        if should_keep:
            kept_records.append(dict(row))

    return pd.DataFrame(kept_records)

def final_consolidation_400m(stops_df, radius_m=400):
    """
    Merge stations <= 400m into one centroid.
    The final DataFrame has columns:
      - consolidated_name
      - lat
      - lon
    plus the cluster’s combined station names + routes in that name.
    """
    clusters = []
    for _, row in stops_df.iterrows():
        s_name = row.get("stop_name", None)
        s_lat  = row.get("stop_lat", None)
        s_lon  = row.get("stop_lon", None)
        s_routes = row.get("routes", set())
        
        found_cluster = False
        for cl in clusters:
            dist = haversine_m(s_lat, s_lon, cl["lat"], cl["lon"])
            if dist <= radius_m:
                # Merge
                cl["names"].append(s_name)
                cl["routes"].update(s_routes)
                cl["all_lats"].append(s_lat)
                cl["all_lons"].append(s_lon)
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
    
    records = []
    for cl in clusters:
        all_names = sorted(set(cl["names"]))
        combined_name = " / ".join(str(n) for n in all_names if n)
        
        route_list = sorted(cl["routes"])
        if route_list:
            combined_name += f" ({', '.join(route_list)})"
        
        n_stations = len(all_names)
        if n_stations > 1:
            combined_name += f" ({n_stations} stations)"
        
        records.append({
            "consolidated_name": combined_name,
            "lat": cl["lat"],
            "lon": cl["lon"],
        })
    
    return pd.DataFrame(records)

def generate_kml(final_df, output_file):
    """Write consolidated <Placemark> data to KML."""
    kml_header = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
"""
    kml_footer = """</Document>
</kml>
"""
    placemarks = []
    for _, row in final_df.iterrows():
        name_raw = row["consolidated_name"] or "Unnamed"
        name = name_raw.replace("&", "&amp;")
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
    ########################################################
    # A) Load GTFS
    ########################################################
    gtfs_bs = load_gtfs(SEPTA_BUS_SUBWAY_GTFS_PATH)
    gtfs_rr = load_gtfs(SEPTA_REGIONAL_RAIL_GTFS_PATH)
    gtfs_patco = load_gtfs(PATCO_GTFS_PATH)

    # B) Weekday service IDs
    bs_sids = get_weekday_service_ids(gtfs_bs["calendar"], gtfs_bs["calendar_dates"])
    rr_sids = get_weekday_service_ids(gtfs_rr["calendar"], gtfs_rr["calendar_dates"])
    patco_sids = get_weekday_service_ids(gtfs_patco["calendar"], gtfs_patco["calendar_dates"])

    ########################################################
    # C) BUS FREQUENCY FILTER
    ########################################################
    bus_stops_qualified = filter_bus_stops_by_frequency(
        gtfs_bs, BUS_FREQUENCY, START_TIME, END_TIME, bs_sids
    )

    ########################################################
    # D) NON-BUS STOPS (Subway) in the Bus/Subway feed
    ########################################################
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

    bs_all_stops = bus_stops_qualified.union(non_bus_stops)

    ########################################################
    # E) Regional Rail & PATCO => ALL stops
    ########################################################
    rr_stops_df = gtfs_rr["stops"]
    rr_all_stops = set(rr_stops_df["stop_id"].unique()) if not rr_stops_df.empty else set()

    patco_stops_df = gtfs_patco["stops"]
    patco_all_stops = set(patco_stops_df["stop_id"].unique()) if not patco_stops_df.empty else set()

    ########################################################
    # F) Build Full Data: Bus/Subway feed
    ########################################################
    bs_stops_df = gtfs_bs["stops"]
    if not bs_stops_df.empty:
        bs_filtered = bs_stops_df[bs_stops_df["stop_id"].isin(bs_all_stops)].copy()
        bs_stop_routes_df = get_stop_routes(gtfs_bs)
        bs_filtered = bs_filtered.merge(bs_stop_routes_df, on="stop_id", how="left")
        bs_filtered["routes"] = bs_filtered["routes"].apply(lambda x: x if isinstance(x, set) else set())
        bs_filtered["agency"] = "SEPTA_BusSubway"
    else:
        bs_filtered = pd.DataFrame(columns=["stop_id","stop_name","stop_lat","stop_lon","routes","agency"])

    # 2) Regional Rail
    if not rr_stops_df.empty:
        rr_stop_routes_df = get_stop_routes(gtfs_rr)
        rr_filtered = rr_stops_df[rr_stops_df["stop_id"].isin(rr_all_stops)].copy()
        rr_filtered = rr_filtered.merge(rr_stop_routes_df, on="stop_id", how="left")
        rr_filtered["routes"] = rr_filtered["routes"].apply(lambda x: x if isinstance(x, set) else set())
        rr_filtered["agency"] = "SEPTA_RegionalRail"
    else:
        rr_filtered = pd.DataFrame(columns=["stop_id","stop_name","stop_lat","stop_lon","routes","agency"])

    # 3) PATCO
    if not patco_stops_df.empty:
        patco_stop_routes_df = get_stop_routes(gtfs_patco)
        patco_filtered = patco_stops_df[patco_stops_df["stop_id"].isin(patco_all_stops)].copy()
        patco_filtered = patco_filtered.merge(patco_stop_routes_df, on="stop_id", how="left")
        patco_filtered["routes"] = patco_filtered["routes"].apply(lambda x: x if isinstance(x, set) else set())
        patco_filtered["agency"] = "PATCO"
    else:
        patco_filtered = pd.DataFrame(columns=["stop_id","stop_name","stop_lat","stop_lon","routes","agency"])

    ########################################################################
    # G) Filter bus stops that are <250m from a kept stop (except <100m+multiline)
    ########################################################################
    candidate_bus = bs_filtered[bs_filtered["stop_id"].isin(bus_stops_qualified)].copy()
    nonbus_part   = bs_filtered[~bs_filtered["stop_id"].isin(bus_stops_qualified)].copy()

    filtered_bus = filter_close_bus_stops(
        candidate_bus,
        dist_remove=BUS_TOO_CLOSE_RADIUS_M,
        dist_multiline=BUS_MULTI_LINE_RADIUS_M
    )

    bs_filtered_final = pd.concat([filtered_bus, nonbus_part], ignore_index=True)

    ########################################################
    # H) Combine all feeds
    ########################################################
    all_stops = pd.concat([bs_filtered_final, rr_filtered, patco_filtered], ignore_index=True)

    # Make sure we have needed columns
    for col in ["stop_name","stop_lat","stop_lon","routes"]:
        if col not in all_stops.columns:
            all_stops[col] = None

    ########################################################
    # I) Final step: Consolidate within 400m
    ########################################################
    final_df = final_consolidation_400m(all_stops, radius_m=FINAL_CONSOLIDATION_RADIUS_M)

    ########################################################
    # J) Generate KML
    ########################################################
    generate_kml(final_df, OUTPUT_KML)
    print(f"KML saved to: {OUTPUT_KML}")

    ########################################################
    # K) ALSO EXPORT stations.csv FOR RING-CREATION SCRIPT
    ########################################################
    # We'll give each consolidated point a "stop_id" like cluster_0, cluster_1, ...
    final_df["stop_id"] = [f"cluster_{i}" for i in final_df.index]
    final_df["stop_name"] = final_df["consolidated_name"]
    final_df["stop_lat"] = final_df["lat"]
    final_df["stop_lon"] = final_df["lon"]

    # We'll keep these 4 columns. The ring script usually needs at least
    # stop_lat, stop_lon, plus possibly an ID or name.
    final_df[["stop_id","stop_name","stop_lat","stop_lon"]].to_csv(OUTPUT_CSV, index=False)
    print(f"CSV of final stations saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
