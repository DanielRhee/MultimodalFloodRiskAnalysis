import numpy as np
import config

def calculateElevationRisk(elevation, validMask):
    print("Calculating elevation-based flood risk...")

    elevationRisk = np.zeros_like(elevation, dtype=np.float32)

    for i in range(len(config.ELEVATION_BREAKPOINTS) - 1):
        lowerBound = config.ELEVATION_BREAKPOINTS[i][1]
        upperBound = config.ELEVATION_BREAKPOINTS[i + 1][0]
        lowerRisk = config.ELEVATION_BREAKPOINTS[i][2]
        upperRisk = config.ELEVATION_BREAKPOINTS[i + 1][2]

        mask = (elevation >= lowerBound) & (elevation < upperBound) & validMask

        if np.any(mask):
            elevRange = upperBound - lowerBound
            if elevRange > 0 and not np.isinf(elevRange):
                normalizedElev = (elevation[mask] - lowerBound) / elevRange
                elevationRisk[mask] = lowerRisk + normalizedElev * (upperRisk - lowerRisk)
            else:
                elevationRisk[mask] = lowerRisk

    belowSeaLevel = (elevation < 0) & validMask
    elevationRisk[belowSeaLevel] = 1.0

    above50m = (elevation >= 50) & validMask
    if np.any(above50m):
        elevAbove50 = elevation[above50m]
        elevationRisk[above50m] = 0.1 * np.exp(-(elevAbove50 - 50) / 50)

    print(f"Elevation risk range: [{elevationRisk[validMask].min():.4f}, {elevationRisk[validMask].max():.4f}]")

    return elevationRisk
