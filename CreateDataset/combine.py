import numpy as np
import config

def combineFinalRiskScores(proximityRisk, elevationRisk, landCoverRisk, precipitationFactor, validMask):
    print("\nCombining risk factors into final flood risk scores...")

    riskScore = (
        config.FINAL_WEIGHTS['proximity'] * proximityRisk +
        config.FINAL_WEIGHTS['elevation'] * elevationRisk +
        config.FINAL_WEIGHTS['landcover'] * landCoverRisk +
        config.FINAL_WEIGHTS['precipitation'] * precipitationFactor
    )

    floodRisk = np.clip(riskScore * 100, 0, 100).astype(np.uint8)

    floodRisk[~validMask] = 0

    print(f"\nFinal weights:")
    print(f"  Proximity:     {config.FINAL_WEIGHTS['proximity']:.2f}")
    print(f"  Elevation:     {config.FINAL_WEIGHTS['elevation']:.2f}")
    print(f"  Land cover:    {config.FINAL_WEIGHTS['landcover']:.2f}")
    print(f"  Precipitation: {config.FINAL_WEIGHTS['precipitation']:.2f}")

    validRisks = floodRisk[validMask]
    print(f"\nComponent contributions (mean values):")
    print(f"  Proximity:     {(config.FINAL_WEIGHTS['proximity'] * proximityRisk[validMask].mean() * 100):.2f}")
    print(f"  Elevation:     {(config.FINAL_WEIGHTS['elevation'] * elevationRisk[validMask].mean() * 100):.2f}")
    print(f"  Land cover:    {(config.FINAL_WEIGHTS['landcover'] * landCoverRisk[validMask].mean() * 100):.2f}")
    print(f"  Precipitation: {(config.FINAL_WEIGHTS['precipitation'] * precipitationFactor * 100):.2f}")

    return floodRisk
