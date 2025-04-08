import os
import zipfile
import pandas as pd

###############################################################################
# CONFIG
###############################################################################

# Paths to your three GTFS pools:
SEPTA_BUS_SUBWAY_GTFS_PATH = "./gtfs/SEPTABus"  # or e.g., "/path/to/septa_bus_subway.zip"
SEPTA_REGIONAL_RAIL_GTFS_PATH = "./gtfs/SEPTARail"
PATCO_GTFS_PATH = "./gtfs/PATCO"

# The consolidated CSV that has 'original_stop_ids' column, e.g. "300,301"
STATIONS_CSV = "./out/stationsplus.csv"

# Output KMLs
REGIONAL_RAIL_KML = "./out/lines/regional_rail.kml"
BUS_KML = "./out/lines/bus.kml"
OTHER_KML = "./out/lines/other.kml"

###############################################################################
# 1) Load GTFS
###############################################################################

def load_gtfs(gtfs_path):
    """
    Load relevant GTFS files (routes, trips, stop_times, shapes)
    from a directory or a .zip file. Returns a dict of DataFrames.
    """
    def read_csv_from_dir(directory, filename):
        fp = os.path.join(directory, filename)
        if not os.path.exists(fp):
            return pd.DataFrame()
        return pd.read_csv(fp)
    
    def read_csv_from_zip(zip_path, filename):
        with zipfile.ZipFile(zip_path, 'r') as z:
            if filename not in z.namelist():
                return pd.DataFrame()
            with z.open(filename) as f:
                return pd.read_csv(f)
    
    if os.path.isdir(gtfs_path):
        reader = lambda fname: read_csv_from_dir(gtfs_path, fname)
    else:
        reader = lambda fname: read_csv_from_zip(gtfs_path, fname)
    
    routes_df     = reader("routes.txt")
    trips_df      = reader("trips.txt")
    stop_times_df = reader("stop_times.txt")
    shapes_df     = reader("shapes.txt")
    
    # IMPORTANT FIX: Convert stop_id column to string if present
    if "stop_id" in stop_times_df.columns:
        stop_times_df["stop_id"] = stop_times_df["stop_id"].astype(str)
    
    return {
        "routes": routes_df,
        "trips": trips_df,
        "stop_times": stop_times_df,
        "shapes": shapes_df
    }

###############################################################################
# 2) Build shapes dictionary
###############################################################################

def build_shapes_dict(shapes_df):
    """
    Creates a dict: shape_id -> list of (lat, lon) in shape_pt_sequence order.
    Returns empty if shapes.txt is missing or doesn't have needed columns.
    """
    if shapes_df.empty:
        return {}
    
    needed_cols = {"shape_id","shape_pt_lat","shape_pt_lon","shape_pt_sequence"}
    if not needed_cols.issubset(shapes_df.columns):
        return {}
    
    shapes_df = shapes_df.dropna(subset=needed_cols)
    shapes_df["shape_pt_sequence"] = shapes_df["shape_pt_sequence"].astype(int)
    
    shapes_dict = {}
    for sid, grp in shapes_df.groupby("shape_id"):
        grp = grp.sort_values("shape_pt_sequence")
        coords = list(zip(grp["shape_pt_lat"], grp["shape_pt_lon"]))
        shapes_dict[sid] = coords
    return shapes_dict

###############################################################################
# 3) route_id -> shape_ids
###############################################################################

def build_route_shapes_dict(trips_df):
    """
    Return a dict: route_id -> set of shape_ids for that route.
    """
    if trips_df.empty or "shape_id" not in trips_df.columns:
        return {}
    
    needed = {"route_id","shape_id"}
    df = trips_df.dropna(subset=needed)
    
    route_shapes = {}
    for rid, grp in df.groupby("route_id"):
        shape_ids = set(grp["shape_id"].unique())
        route_shapes[rid] = shape_ids
    return route_shapes

###############################################################################
# 4) Make KML lines from shape coords
###############################################################################

def make_kml_for_lines(lines, route_name):
    """
    lines: list of lists of (lat,lon), each sub-list is one shape
    route_name: string
    Returns a KML string of <Placemark> items.
    """
    kml_fragments = []
    for i, coords in enumerate(lines):
        if not coords:
            continue
        coord_str = " ".join(f"{lon},{lat},0" for (lat, lon) in coords)
        label = route_name if i == 0 else f"{route_name} (part {i+1})"
        placemark = f"""
  <Placemark>
    <name>{label}</name>
    <Style>
      <LineStyle>
        <color>ff0000ff</color> <!-- Red in ABGR for KML -->
        <width>3</width>
      </LineStyle>
    </Style>
    <LineString>
      <coordinates>{coord_str}</coordinates>
    </LineString>
  </Placemark>
"""
        kml_fragments.append(placemark)
    return "".join(kml_fragments)

def write_kml_file(placemarks_str, filename):
    """Write the full KML doc to 'filename'."""
    kml_header = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
"""
    kml_footer = """</Document>
</kml>
"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(kml_header)
        f.write(placemarks_str)
        f.write(kml_footer)

###############################################################################
# 5) For each feed, generate KML lines for (rr, bus, other)
###############################################################################

def extract_feed_placemarks(gtfs, valid_stop_ids):
    """
    Given one GTFS feed (dict of DataFrames) and a set of valid_stop_ids (strings),
    returns (rr_kml, bus_kml, other_kml) where each is a string of placemarks.
    
    We do the standard approach:
      1) stop_times -> filter by valid_stop_ids
      2) find those trips
      3) find routes
      4) route_type=2 => RR, 3 => bus, else => other
      5) build shapes => line placemarks
    """
    routes_df = gtfs["routes"]
    trips_df  = gtfs["trips"]
    st_times  = gtfs["stop_times"]
    shapes_df = gtfs["shapes"]
    
    if routes_df.empty or trips_df.empty or st_times.empty:
        return ("","","")

    # subset st_times by valid_stop_ids
    subset_st = st_times[st_times["stop_id"].isin(valid_stop_ids)]
    if subset_st.empty:
        return ("","","")
    
    good_trip_ids = set(subset_st["trip_id"].unique())
    sub_trips = trips_df[trips_df["trip_id"].isin(good_trip_ids)]
    if sub_trips.empty:
        return ("","","")
    
    good_route_ids = set(sub_trips["route_id"].unique())
    sub_routes = routes_df[routes_df["route_id"].isin(good_route_ids)].copy()
    if "route_type" not in sub_routes.columns:
        sub_routes["route_type"] = -1
    
    shapes_dict = build_shapes_dict(shapes_df)
    route_shapes_map = build_route_shapes_dict(sub_trips)
    
    rr_kml    = []
    bus_kml   = []
    other_kml = []
    
    for _, rrow in sub_routes.iterrows():
        rtype = int(rrow["route_type"])
        rid = rrow["route_id"]
        rname = rrow.get("route_short_name") or rrow.get("route_long_name") or f"Route {rid}"
        
        shape_ids = route_shapes_map.get(rid, set())
        if not shape_ids:
            continue
        
        lines = []
        for sid in shape_ids:
            coords = shapes_dict.get(sid, [])
            if coords:
                lines.append(coords)
        
        if not lines:
            continue
        
        # Make placemarks
        route_pm = make_kml_for_lines(lines, rname)
        
        if rtype == 2:
            rr_kml.append(route_pm)
        elif rtype == 3:
            bus_kml.append(route_pm)
        else:
            other_kml.append(route_pm)
    
    return ("".join(rr_kml), "".join(bus_kml), "".join(other_kml))

###############################################################################
# 6) Main
###############################################################################

def main():
    # A) Read stations.csv => parse original_stop_ids
    stations_df = pd.read_csv(STATIONS_CSV)
    
    if "original_stop_ids" not in stations_df.columns:
        print(f"Error: {STATIONS_CSV} missing 'original_stop_ids' column.")
        return
    
    # Combine all real stop IDs from CSV
    all_real_stop_ids = set()
    for _, row in stations_df.iterrows():
        raw_str = str(row["original_stop_ids"])
        splitted = [x.strip() for x in raw_str.split(",") if x.strip()]
        all_real_stop_ids.update(splitted)
    
    if not all_real_stop_ids:
        print("No real stop IDs found in stations.csv. Exiting.")
        return
    
    # B) Load each GTFS pool
    gtfs_bs    = load_gtfs(SEPTA_BUS_SUBWAY_GTFS_PATH)
    gtfs_rr    = load_gtfs(SEPTA_REGIONAL_RAIL_GTFS_PATH)
    gtfs_patco = load_gtfs(PATCO_GTFS_PATH)

    # C) For each GTFS feed, gather route lines
    #    We'll combine them in 3 global strings (RR, Bus, Other)
    global_rr_kml    = []
    global_bus_kml   = []
    global_other_kml = []

    # Bus/Subway feed
    rr_kml, bus_kml, oth_kml = extract_feed_placemarks(gtfs_bs, all_real_stop_ids)
    global_rr_kml.append(rr_kml)
    global_bus_kml.append(bus_kml)
    global_other_kml.append(oth_kml)

    # Regional Rail feed
    rr_kml, bus_kml, oth_kml = extract_feed_placemarks(gtfs_rr, all_real_stop_ids)
    global_rr_kml.append(rr_kml)
    global_bus_kml.append(bus_kml)
    global_other_kml.append(oth_kml)

    # PATCO feed
    rr_kml, bus_kml, oth_kml = extract_feed_placemarks(gtfs_patco, all_real_stop_ids)
    global_rr_kml.append(rr_kml)
    global_bus_kml.append(bus_kml)
    global_other_kml.append(oth_kml)

    final_rr_kml    = "".join(global_rr_kml)
    final_bus_kml   = "".join(global_bus_kml)
    final_other_kml = "".join(global_other_kml)

    # D) Write KML
    if final_rr_kml.strip():
        write_kml_file(final_rr_kml, REGIONAL_RAIL_KML)
        print(f"Regional Rail lines => {REGIONAL_RAIL_KML}")
    else:
        print("No Regional Rail lines found for these stations.")

    if final_bus_kml.strip():
        write_kml_file(final_bus_kml, BUS_KML)
        print(f"Bus lines => {BUS_KML}")
    else:
        print("No Bus lines found for these stations.")

    if final_other_kml.strip():
        write_kml_file(final_other_kml, OTHER_KML)
        print(f"Other lines => {OTHER_KML}")
    else:
        print("No 'Other' lines found for these stations.")

if __name__ == "__main__":
    main()