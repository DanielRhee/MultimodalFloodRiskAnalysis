import numpy as np
import pandas as pd
import config

def calculateCompositeSeverity(df):
    severities = []

    for _, row in df.iterrows():
        components = {}
        weights = {}

        if pd.notna(row.get('DEPTH')):
            depth = min(row['DEPTH'], config.SEVERITY_CAPS['DEPTH'])
            components['DEPTH'] = depth / config.SEVERITY_CAPS['DEPTH']
            weights['DEPTH'] = config.SEVERITY_WEIGHTS['DEPTH']

        if pd.notna(row.get('DAMAGE')):
            damage = min(row['DAMAGE'], config.SEVERITY_CAPS['DAMAGE'])
            components['DAMAGE'] = np.log1p(damage) / np.log1p(config.SEVERITY_CAPS['DAMAGE'])
            weights['DAMAGE'] = config.SEVERITY_WEIGHTS['DAMAGE']

        if pd.notna(row.get('DURATION')):
            duration = min(row['DURATION'], config.SEVERITY_CAPS['DURATION'])
            components['DURATION'] = np.sqrt(duration) / np.sqrt(config.SEVERITY_CAPS['DURATION'])
            weights['DURATION'] = config.SEVERITY_WEIGHTS['DURATION']

        if pd.notna(row.get('FATALITY')) and row['FATALITY'] > 0:
            components['FATALITY'] = min(0.5 + np.log1p(row['FATALITY']) / 10, 1.0)
            weights['FATALITY'] = config.SEVERITY_WEIGHTS['FATALITY']

        if components:
            totalWeight = sum(weights.values())
            normalizedWeights = {k: v/totalWeight for k, v in weights.items()}
            severity = sum(components[k] * normalizedWeights[k] for k in components)
        else:
            severity = config.BASELINE_SEVERITY

        severities.append(severity)

    return np.array(severities)

def processFloodData(df):
    print("Calculating composite severities...")
    df['SEVERITY_COMPOSITE'] = calculateCompositeSeverity(df)

    validFloodMask = pd.notna(df['LAT']) & pd.notna(df['LON'])
    validFloodDf = df[validFloodMask].copy()

    print(f"Total flood records: {len(df)}")
    print(f"Valid flood records with coordinates: {len(validFloodDf)}")
    print(f"Severity range: [{validFloodDf['SEVERITY_COMPOSITE'].min():.3f}, {validFloodDf['SEVERITY_COMPOSITE'].max():.3f}]")

    return validFloodDf
