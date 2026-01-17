import numpy as np
import time

import utils
import severity
import proximity
import elevation
import landcover
import precipitation
import combine

def main():
    startTime = time.time()

    print("="*60)
    print("FLOOD RISK DATASET CREATION")
    print("="*60)

    print("\n[1/8] Loading aligned data...")
    alignedData = utils.loadAlignedData()
    rgb = alignedData['rgb']
    elevationData = alignedData['elevation']
    validMask = alignedData['validMask']
    transform = alignedData['transform']
    crs = alignedData['crs']

    print(f"Data shape: {rgb.shape}")
    print(f"Valid pixels: {validMask.sum():,} / {validMask.size:,} ({100*validMask.sum()/validMask.size:.2f}%)")

    print("\n[2/8] Loading flood history data...")
    floodDf = utils.loadFloodHistory()
    validFloodDf = severity.processFloodData(floodDf)

    print("\n[3/8] Building spatial index for flood locations...")
    kdTree, floodPoints, floodSeverities = proximity.buildFloodKDTree(validFloodDf)

    print("\n[4/8] Calculating proximity-based risk...")
    proximityRisk = proximity.calculateProximityRisk(
        shape=rgb.shape[:2],
        transform=transform,
        kdTree=kdTree,
        floodSeverities=floodSeverities,
        validMask=validMask
    )

    print("\n[5/8] Calculating elevation-based risk...")
    elevationRisk = elevation.calculateElevationRisk(elevationData, validMask)

    print("\n[6/8] Loading land cover model and calculating land cover risk...")
    model = landcover.loadLandCoverModel()
    landCoverRisk = landcover.calculateLandCoverRisk(rgb, validMask, model)

    print("\n[7/8] Calculating precipitation factor...")
    precipitationFactor = precipitation.calculatePrecipitationFactor()

    print("\n[8/8] Combining all risk factors...")
    floodRisk = combine.combineFinalRiskScores(
        proximityRisk=proximityRisk,
        elevationRisk=elevationRisk,
        landCoverRisk=landCoverRisk,
        precipitationFactor=precipitationFactor,
        validMask=validMask
    )

    metadata = {
        'description': 'Flood risk scores (0-100) for SF Bay Area at 10m resolution',
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
        'weights': combine.config.FINAL_WEIGHTS,
        'flood_season_months': combine.config.FLOOD_SEASON_MONTHS,
        'precipitation_factor': float(precipitationFactor)
    }

    print("\n" + "="*60)
    utils.printStatistics(floodRisk, validMask)
    print("="*60)

    print("\nSaving dataset...")
    utils.saveFloodRiskDataset(floodRisk, validMask, transform, crs, metadata)

    elapsedTime = time.time() - startTime
    print(f"\nTotal processing time: {elapsedTime:.1f} seconds ({elapsedTime/60:.1f} minutes)")
    print("\nDone!")

if __name__ == '__main__':
    main()
