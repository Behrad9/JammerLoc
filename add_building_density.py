import os
import time
import numpy as np
import pandas as pd
import osmnx as ox
from tqdm import tqdm

# -------------------------------------------------------------------
# OSMnx Configuration
# -------------------------------------------------------------------
ox.settings.timeout = 600
ox.settings.use_cache = True
ox.settings.log_console = False


def get_building_count(lat, lon, radius=100):
    """
    Query OpenStreetMap for building count at a specific location.
    
    Parameters:
    - lat: Latitude
    - lon: Longitude
    - radius: Search radius in meters (default: 100m)
    
    Returns:
    - Number of buildings found
    """
    max_retries = 2
    endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ]
    
    for endpoint in endpoints:
        ox.settings.overpass_endpoint = endpoint
        for attempt in range(max_retries):
            try:
                buildings = ox.features_from_point(
                    center_point=(lat, lon),
                    dist=radius,
                    tags={'building': True}
                )
                return len(buildings)
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                continue
    return 0


def calculate_density_grid(df, grid_size=0.0005, radius=50):
    """
    Calculate building density using a grid-based approach.
    This reduces API calls by grouping nearby points.
    
    Parameters:
    - df: DataFrame with 'lat' and 'lon' columns
    - grid_size: Grid cell size in degrees (default: ~55m at mid-latitudes)
    - radius: Search radius in meters for each grid cell
    
    Returns:
    - DataFrame with added 'building_density' column (buildings/km²)
    """
    print(f"Creating grid with cell size: {grid_size} degrees (~{grid_size*111000:.0f}m)")
    print(f"Using search radius: {radius} meters")
    
    df = df.copy()
    
    # Create grid coordinates
    df['grid_lat'] = (df['lat'] / grid_size).round() * grid_size
    df['grid_lon'] = (df['lon'] / grid_size).round() * grid_size
    
    # Get unique grid cells
    unique_grids = df[['grid_lat', 'grid_lon']].drop_duplicates()
    print(f"Found {len(unique_grids)} unique grid cells")
    
    # Calculate density for each grid cell
    grid_densities = {}
    failed_queries = 0
    
    print("Querying OpenStreetMap for building counts...")
    for _, grid in tqdm(unique_grids.iterrows(), total=len(unique_grids)):
        try:
            building_count = get_building_count(
                grid['grid_lat'],
                grid['grid_lon'],
                radius
            )
            
            # Calculate density (buildings per km²)
            area_km2 = (np.pi * radius**2) / 1_000_000
            density = building_count / area_km2 if area_km2 > 0 else 0
            
            grid_densities[(grid['grid_lat'], grid['grid_lon'])] = density
            
        except Exception as e:
            failed_queries += 1
            grid_densities[(grid['grid_lat'], grid['grid_lon'])] = 0.0
    
    if failed_queries > 0:
        print(f"Warning: {failed_queries} grid cells failed, set to density=0")
    
    # Map densities back to original points
    df['building_density'] = df.apply(
        lambda row: grid_densities.get((row['grid_lat'], row['grid_lon']), 0.0),
        axis=1
    )
    
    # Clean up temporary columns
    df = df.drop(['grid_lat', 'grid_lon'], axis=1)
    
    return df


def calculate_density_direct(df, radius=50):
    """
    Calculate building density by querying each point individually.
    More accurate but slower - use for small datasets.
    
    Parameters:
    - df: DataFrame with 'lat' and 'lon' columns
    - radius: Search radius in meters
    
    Returns:
    - DataFrame with added 'building_density' column (buildings/km²)
    """
    print(f"Calculating building density for {len(df)} points...")
    print(f"Using search radius: {radius} meters")
    
    df = df.copy()
    densities = []
    
    area_km2 = (np.pi * radius**2) / 1_000_000
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        building_count = get_building_count(row['lat'], row['lon'], radius)
        density = building_count / area_km2 if area_km2 > 0 else 0
        densities.append(density)
    
    df['building_density'] = densities
    
    return df


def add_building_density(csv_path, output_path=None, method='direct', 
                         grid_size=0.0005, radius=50):
    """
    Main function to add building density to a CSV file.
    
    Parameters:
    - csv_path: Path to input CSV file (must have 'lat' and 'lon' columns)
    - output_path: Path to save output CSV (default: adds '_with_density' suffix)
    - method: 'grid' (faster, recommended) or 'direct' (more accurate)
    - grid_size: Grid cell size in degrees (only for 'grid' method)
    - radius: Search radius in meters
    
    Returns:
    - DataFrame with building density added
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    if 'lat' not in df.columns or 'lon' not in df.columns:
        raise ValueError("CSV must contain 'lat' and 'lon' columns")
    
    print(f"Loaded {len(df)} records")
    print(f"\nGeographic bounds:")
    print(f"  Latitude:  {df['lat'].min():.6f} to {df['lat'].max():.6f}")
    print(f"  Longitude: {df['lon'].min():.6f} to {df['lon'].max():.6f}")
    
    # Calculate building density
    if method == 'grid':
        df = calculate_density_grid(df, grid_size=grid_size, radius=radius)
    elif method == 'direct':
        df = calculate_density_direct(df, radius=radius)
    else:
        raise ValueError("method must be 'grid' or 'direct'")
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"BUILDING DENSITY STATISTICS")
    print(f"{'='*60}")
    print(f"Min:     {df['building_density'].min():.0f} buildings/km²")
    print(f"Max:     {df['building_density'].max():.0f} buildings/km²")
    print(f"Mean:    {df['building_density'].mean():.0f} buildings/km²")
    print(f"Median:  {df['building_density'].median():.0f} buildings/km²")
    print(f"Std Dev: {df['building_density'].std():.0f} buildings/km²")
    
    non_zero = (df['building_density'] > 0).sum()
    print(f"\nNon-zero density: {non_zero}/{len(df)} ({non_zero/len(df)*100:.1f}%)")
    
    # Save output
    if output_path is None:
        base, ext = os.path.splitext(csv_path)
        output_path = f"{base}_with_density{ext}"
    
    df.to_csv(output_path, index=False)
    print(f"\nOutput saved to: {output_path}")
    
    return df


# -------------------------------------------------------------------
# Example Usage
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Example 1: Using grid method (recommended for large datasets)
    INPUT_FILE = "trajectories_for_osm.csv"
    OUTPUT_FILE = "trajectories_for_osm_with_density.csv"
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: '{INPUT_FILE}' not found.")
        print("\nCreating example CSV file...")
        
        # Create example data (Turin, Italy area)
        example_data = pd.DataFrame({
            'lat': [45.0645, 45.0650, 45.0655],
            'lon': [7.6585, 7.6590, 7.6595],
            'location': ['Point 1', 'Point 2', 'Point 3']
        })
        example_data.to_csv(INPUT_FILE, index=False)
        print(f"Created example file: {INPUT_FILE}")
    
    try:
        # Method 1: Grid-based (faster, good for many points)
        # result = add_building_density(
           #  INPUT_FILE,
          #   OUTPUT_FILE,
        #    method='grid',      # Use grid for efficiency
        #    grid_size=0.0005,   # ~55m grid cells
          #  radius=100          # 100m search radius
        #)
        
        # Method 2: Direct calculation (uncomment to use)
        result = add_building_density(
            INPUT_FILE,
            OUTPUT_FILE, 
           method='direct',
           radius=50
        )
        
        print("\n Processing completed successfully!")
        
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()