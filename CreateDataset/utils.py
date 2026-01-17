import numpy as np
import pandas as pd
import config

def loadAlignedData():
    data = np.load(config.ALIGNED_DATA_PATH)
    rgb = data['rgb']
    if rgb.shape[0] == 3:
        rgb = np.transpose(rgb, (1, 2, 0))
    return {
        'rgb': rgb,
        'elevation': data['elevation'],
        'validMask': data['validMask'],
        'transform': data['transform'],
        'crs': str(data['crs'])
    }

def loadFloodHistory():
    df = pd.read_csv(config.FLOOD_HISTORY_PATH, low_memory=False)
    return df

def saveFloodRiskDataset(floodRisk, validMask, transform, crs, metadata):
    np.savez_compressed(
        config.OUTPUT_PATH,
        floodRisk=floodRisk.astype(np.uint8),
        validMask=validMask,
        transform=transform,
        crs=crs,
        metadata=metadata
    )
    print(f"Saved flood risk dataset to {config.OUTPUT_PATH}")

def printStatistics(floodRisk, validMask):
    validRisks = floodRisk[validMask]
    print("\n=== Flood Risk Statistics ===")
    print(f"Total valid pixels: {validRisks.size:,}")
    print(f"Min risk: {validRisks.min()}")
    print(f"Max risk: {validRisks.max()}")
    print(f"Mean risk: {validRisks.mean():.2f}")
    print(f"Median risk: {np.median(validRisks):.2f}")
    print(f"Std risk: {validRisks.std():.2f}")

    print("\n=== Risk Distribution ===")
    bins = [0, 20, 40, 60, 80, 100]
    labels = ['Very Low (0-20)', 'Low (20-40)', 'Medium (40-60)', 'High (60-80)', 'Very High (80-100)']
    for i in range(len(bins)-1):
        count = np.sum((validRisks >= bins[i]) & (validRisks < bins[i+1]))
        pct = 100 * count / validRisks.size
        print(f"{labels[i]}: {count:,} ({pct:.2f}%)")
