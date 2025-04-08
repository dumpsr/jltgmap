import pandas as pd
import pyproj
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union, transform
import math

###########################################################
# 1) User Configuration
###########################################################

INPUT_CSV = "./out/stations.csv"   # Example. Or replace with your own approach to loading stops.
OUTPUT_KML = "./out/stations_donut_800m_2km.kml"

# Distances for the donut
INNER_DISTANCE = 0 # meters
OUTER_DISTANCE = 800 # meters

# Choose a projected coordinate system (CRS) that is suitable for your region.
# Example: UTM zone 18N (EPSG:32618). Adjust to your actual area.
LOCAL_CRS = "EPSG:32618"

###########################################################
# 2) Helper Functions
###########################################################

def load_stations(csv_file):
    """
    Reads a CSV of stations with columns: stop_id, stop_lat, stop_lon.
    Returns a pandas DataFrame.
    """
    df = pd.read_csv(csv_file)
    return df

def latlon_to_local_transformer(local_crs=LOCAL_CRS):
    """
    Returns a pair of transformer functions:
      - forward: (lon, lat) -> (x, y) in local projected coords
      - inverse: (x, y) -> (lon, lat)
    We assume input lat/lon is EPSG:4326 (WGS84).
    """
    wgs84 = pyproj.CRS("EPSG:4326")
    local = pyproj.CRS(local_crs)
    
    forward_transformer = pyproj.Transformer.from_crs(wgs84, local, always_xy=True).transform
    inverse_transformer = pyproj.Transformer.from_crs(local, wgs84, always_xy=True).transform
    return forward_transformer, inverse_transformer

def create_donut_geometry(lat, lon, forward_func, inner_dist=800, outer_dist=2000):
    """
    Creates a donut (ring) geometry: all points between inner_dist and outer_dist from (lat, lon).
    - (lat, lon) in WGS84
    - forward_func: function to project (lon, lat)->(x, y)
    """
    # Note shapely wants (x, y) => (east, north)
    x, y = forward_func(lon, lat)
    
    center = Point(x, y)
    outer_circle = center.buffer(outer_dist)  # outer buffer
    inner_circle = center.buffer(inner_dist)  # inner buffer
    donut = outer_circle.difference(inner_circle)
    return donut

###########################################################
# 3) KML Export
###########################################################

def polygon_to_kml_coords(polygon):
    """
    Given a shapely Polygon in WGS84, return a string of <outerboundary> + <innerboundary> (if any) in KML format.
    Each ring is "lon,lat,0" repeated.
    """
    # polygon.exterior => outer boundary
    # polygon.interiors => list of holes
    # We assume the polygon is already in lat/lon.
    
    def ring_to_kml(ring):
        coords = []
        for lon, lat in ring.coords:
            coords.append(f"{lon},{lat},0")
        # Must close the ring (last = first)
        # Shapely's ring is already closed, so it's repeated. We'll just join them.
        return " ".join(coords)
    
    # Outer boundary
    outer_str = ring_to_kml(polygon.exterior)
    outer_kml = f"""
      <outerBoundaryIs>
        <LinearRing>
          <coordinates>{outer_str}</coordinates>
        </LinearRing>
      </outerBoundaryIs>
    """
    
    # Inner boundaries (holes)
    inner_kml = ""
    for hole in polygon.interiors:
        hole_str = ring_to_kml(hole)
        inner_kml += f"""
      <innerBoundaryIs>
        <LinearRing>
          <coordinates>{hole_str}</coordinates>
        </LinearRing>
      </innerBoundaryIs>
        """
    
    return outer_kml + inner_kml

def multipolygon_to_kml(mpoly):
    """
    Convert a shapely (Multi)Polygon in lat/lon (EPSG:4326) to KML <Polygon> elements.
    Returns a string with multiple <Polygon> elements if needed.
    """
    polygons = []
    if isinstance(mpoly, Polygon):
        polygons = [mpoly]
    elif isinstance(mpoly, MultiPolygon):
        polygons = list(mpoly.geoms)
    else:
        # Not a polygon at all
        return ""
    
    polygon_kml_fragments = []
    for poly in polygons:
        rings_kml = polygon_to_kml_coords(poly)
        poly_kml = f"""
    <Placemark>
      <name>Donut Zone</name>
      <Style>
        <PolyStyle>
          <color>7f00ff00</color>   <!-- A semi-transparent green -->
          <outline>1</outline>
        </PolyStyle>
      </Style>
      <Polygon>
        <tessellate>1</tessellate>
        {rings_kml}
      </Polygon>
    </Placemark>
"""
        polygon_kml_fragments.append(poly_kml)
    
    return "\n".join(polygon_kml_fragments)

def write_kml_polygon(shape_wgs84, out_file):
    """
    Write the (Multi)Polygon shape in WGS84 to a KML file.
    """
    kml_header = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
"""
    kml_footer = """
</Document>
</kml>
"""
    # Convert shape to KML polygons
    polygons_kml = multipolygon_to_kml(shape_wgs84)
    
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(kml_header)
        f.write(polygons_kml)
        f.write(kml_footer)


###########################################################
# 4) Main Logic
###########################################################
def main():
    # A) Load stations
    stops_df = load_stations(INPUT_CSV)  # or build your DataFrame however you like
    
    # B) Build transformer to local coords, and an inverse back to WGS84
    forward_func, inverse_func = latlon_to_local_transformer(LOCAL_CRS)
    
    # C) Create donuts & union
    donut_list = []
    for _, row in stops_df.iterrows():
        lat = row["stop_lat"]
        lon = row["stop_lon"]
        donut_local = create_donut_geometry(lat, lon, forward_func, INNER_DISTANCE, OUTER_DISTANCE)
        donut_list.append(donut_local)
    
    union_local = unary_union(donut_list)  # union of all donuts in local coords
    
    # D) Transform the unioned shape back to WGS84 for KML
    union_wgs84 = transform(inverse_func, union_local)
    
    # E) Write to KML
    write_kml_polygon(union_wgs84, OUTPUT_KML)
    print(f"Done! Wrote donut polygon to {OUTPUT_KML}")

if __name__ == "__main__":
    main()
