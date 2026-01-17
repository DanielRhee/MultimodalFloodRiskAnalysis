#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from pathlib import Path


def loadAlignedData(npzPath):
    data = np.load(npzPath)
    rgb = data['rgb']
    elevation = data['elevation']
    validMask = data['validMask']

    return rgb, elevation, validMask


def downsampleData(rgb, elevation, validMask, maxDim=2000):
    currentMaxDim = max(rgb.shape[1], rgb.shape[2])

    if currentMaxDim <= maxDim:
        return rgb, elevation, validMask

    scale = maxDim / currentMaxDim

    rgbDownsampled = zoom(rgb, (1, scale, scale), order=1)
    elevationDownsampled = zoom(elevation, scale, order=1)
    maskDownsampled = zoom(validMask.astype(float), scale, order=0) > 0.5

    return rgbDownsampled, elevationDownsampled, maskDownsampled


def createOverviewFigure(rgb, elevation, validMask):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    rgbDisplay = np.transpose(rgb, (1, 2, 0))

    maskInvalid = ~validMask
    rgbDisplay = rgbDisplay.copy()
    rgbDisplay[maskInvalid] = [100, 100, 100]

    ax1.imshow(rgbDisplay)
    ax1.set_title('RGB Imagery (Sentinel-2)', fontsize=14)
    ax1.set_xlabel('Easting (pixels)')
    ax1.set_ylabel('Northing (pixels)')

    elevationMasked = elevation.copy()
    elevationMasked[maskInvalid] = np.nan

    im = ax2.imshow(elevationMasked, cmap='terrain', interpolation='nearest')
    ax2.set_title('Elevation (10m DEM)', fontsize=14)
    ax2.set_xlabel('Easting (pixels)')
    ax2.set_ylabel('Northing (pixels)')

    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Elevation (meters)', rotation=270, labelpad=20)

    validPixels = np.sum(validMask)
    totalPixels = validMask.size
    validPercent = 100 * validPixels / totalPixels

    fig.suptitle(
        f'Aligned RGB-Elevation Data (UTM Zone 10N)\n'
        f'Valid pixels: {validPixels:,} ({validPercent:.1f}%)',
        fontsize=16, y=0.98
    )

    plt.tight_layout()

    return fig


def main():
    basePath = Path(__file__).parent
    npzPath = basePath / "aligned_data.npz"
    outputPath = basePath / "aligned_data_viz.png"

    print("Loading aligned data...")
    rgb, elevation, validMask = loadAlignedData(npzPath)

    print(f"Original shape: RGB {rgb.shape}, Elevation {elevation.shape}")

    print("Downsampling to display resolution...")
    rgbDisplay, elevDisplay, maskDisplay = downsampleData(rgb, elevation, validMask, maxDim=2000)

    print(f"Display shape: RGB {rgbDisplay.shape}, Elevation {elevDisplay.shape}")

    print("Creating visualization...")
    fig = createOverviewFigure(rgbDisplay, elevDisplay, maskDisplay)

    print(f"Saving to {outputPath}...")
    fig.savefig(outputPath, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("\n✓ Visualization complete!")


if __name__ == "__main__":
    main()
