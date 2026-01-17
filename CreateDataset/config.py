import numpy as np

ALIGNED_DATA_PATH = '../aligned_data.npz'
FLOOD_HISTORY_PATH = '../FloodHistory/USFD_v1.1.csv'
OUTPUT_PATH = 'flood_risk_dataset.npz'

TILE_RESOLUTION = 10.0
TARGET_CRS = 'EPSG:32610'
SOURCE_CRS = 'EPSG:4326'

SEVERITY_WEIGHTS = {
    'DEPTH': 0.35,
    'DAMAGE': 0.30,
    'DURATION': 0.20,
    'FATALITY': 0.15
}

SEVERITY_CAPS = {
    'DEPTH': 5.0,
    'DAMAGE': 50_000_000,
    'DURATION': 30.0
}

BASELINE_SEVERITY = 0.2

PROXIMITY_SEARCH_RADIUS = 10000.0
PROXIMITY_SIGMA = 2000.0
PROXIMITY_PERCENTILE_CAP = 95

ELEVATION_BREAKPOINTS = np.array([
    [-np.inf, 0.0, 1.0],
    [0.0, 5.0, 0.8],
    [5.0, 20.0, 0.4],
    [20.0, 50.0, 0.1],
    [50.0, np.inf, 0.0]
])

LANDCOVER_RISK = {
    0: 0.3,
    3: 0.5,
    4: 0.2,
    5: 0.4,
    6: 0.9,
    8: 0.15,
    9: 0.15,
    10: 0.3,
    12: 0.1,
    13: 0.1,
    14: 0.1
}

MONTHLY_PRECIPITATION = {
    1: 114, 2: 118, 3: 82, 4: 38, 5: 18, 6: 4,
    7: 2, 8: 2, 9: 6, 10: 28, 11: 80, 12: 112
}

MAX_PRECIPITATION = 118

FLOOD_SEASON_MONTHS = [11, 12, 1, 2, 3]

FINAL_WEIGHTS = {
    'proximity': 0.45,
    'elevation': 0.30,
    'landcover': 0.15,
    'precipitation': 0.10
}

CHUNK_SIZE = 500
