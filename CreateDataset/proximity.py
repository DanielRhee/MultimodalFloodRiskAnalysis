import numpy as np
from scipy.spatial import cKDTree
from pyproj import Transformer
import config

def buildFloodKDTree(floodDf):
    transformer = Transformer.from_crs(config.SOURCE_CRS, config.TARGET_CRS, always_xy=True)

    lons = floodDf['LON'].values
    lats = floodDf['LAT'].values
    severities = floodDf['SEVERITY_COMPOSITE'].values

    utmX, utmY = transformer.transform(lons, lats)

    floodPoints = np.column_stack([utmX, utmY])
    kdTree = cKDTree(floodPoints)

    print(f"Built KD-tree with {len(floodPoints)} flood points")

    return kdTree, floodPoints, severities

def gaussianKernel(distances, sigma):
    return np.exp(-distances**2 / (2 * sigma**2))

def calculateProximityRisk(shape, transform, kdTree, floodSeverities, validMask):
    height, width = shape
    proximityRisk = np.zeros((height, width), dtype=np.float32)

    print("Calculating proximity-based flood risk...")

    for rowStart in range(0, height, config.CHUNK_SIZE):
        rowEnd = min(rowStart + config.CHUNK_SIZE, height)
        chunkHeight = rowEnd - rowStart

        rowIndices = np.arange(rowStart, rowEnd)[:, None]
        colIndices = np.arange(width)[None, :]

        pixelY = transform[5] + rowIndices * transform[4]
        pixelX = transform[2] + colIndices * transform[0]

        chunkMask = validMask[rowStart:rowEnd]
        validRowIndices, validColIndices = np.where(chunkMask)

        if len(validRowIndices) == 0:
            continue

        queryPoints = np.column_stack([
            pixelX[0, validColIndices],
            pixelY[validRowIndices, 0]
        ])

        indices = kdTree.query_ball_point(queryPoints, config.PROXIMITY_SEARCH_RADIUS)

        for i, neighborIndices in enumerate(indices):
            if len(neighborIndices) == 0:
                continue

            distances = np.linalg.norm(
                queryPoints[i] - kdTree.data[neighborIndices],
                axis=1
            )

            weights = gaussianKernel(distances, config.PROXIMITY_SIGMA)
            severities = floodSeverities[neighborIndices]

            risk = np.sum(weights * severities)

            r = validRowIndices[i]
            c = validColIndices[i]
            proximityRisk[rowStart + r, c] = risk

        if (rowStart // config.CHUNK_SIZE) % 10 == 0:
            print(f"  Processed rows {rowStart}-{rowEnd}/{height}")

    validRisks = proximityRisk[validMask]
    if len(validRisks) > 0:
        capValue = np.percentile(validRisks, config.PROXIMITY_PERCENTILE_CAP)
        proximityRisk = np.clip(proximityRisk, 0, capValue)
        if capValue > 0:
            proximityRisk = proximityRisk / capValue

    print(f"Proximity risk range: [{proximityRisk[validMask].min():.4f}, {proximityRisk[validMask].max():.4f}]")

    return proximityRisk
