import numpy as np
import config

def calculatePrecipitationFactor():
    floodSeasonPrecip = [config.MONTHLY_PRECIPITATION[m] for m in config.FLOOD_SEASON_MONTHS]
    avgFloodSeasonPrecip = np.mean(floodSeasonPrecip)

    precipitationFactor = avgFloodSeasonPrecip / config.MAX_PRECIPITATION

    print(f"Calculating precipitation factor...")
    print(f"Flood season months: {config.FLOOD_SEASON_MONTHS}")
    print(f"Average flood season precipitation: {avgFloodSeasonPrecip:.1f}mm")
    print(f"Precipitation factor: {precipitationFactor:.3f}")

    return precipitationFactor
