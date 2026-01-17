#!/usr/bin/env python3

import os
import glob
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import reproject, Resampling
from rasterio.transform import Affine
from tqdm import tqdm
from pathlib import Path


def loadElevation(ascPath):
    header = {}
    with open(ascPath, 'r') as f:
        for _ in range(6):
            line = f.readline().strip()
            key, value = line.split()
            header[key.lower()] = float(value) if '.' in value else int(value)

    elevationData = np.loadtxt(ascPath, skiprows=6, dtype=np.float32)
    nodata = header.get('nodata_value', -9999)
    elevationData[elevationData == nodata] = np.nan

    return elevationData, header


def buildTargetTransform(metadata):
    cellsize = metadata['cellsize']
    xllcorner = metadata['xllcorner']
    yllcorner = metadata['yllcorner']
    nrows = metadata['nrows']

    yulcorner = yllcorner + (nrows * cellsize)

    transform = Affine(cellsize, 0, xllcorner, 0, -cellsize, yulcorner)

    return transform


def reprojectTileToTarget(tilePath, dstArray, dstTransform, dstCrs):
    with rasterio.open(tilePath) as src:
        for band in range(1, 4):
            reproject(
                source=rasterio.band(src, band),
                destination=dstArray[band-1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dstTransform,
                dst_crs=dstCrs,
                resampling=Resampling.bilinear,
                dst_nodata=0
            )


def computeIntersectionMask(rgbArray, elevationArray):
    rgbValid = np.any(rgbArray > 0, axis=0)
    elevValid = ~np.isnan(elevationArray)
    validMask = rgbValid & elevValid

    return validMask


def saveAlignedData(outputPath, rgbArray, elevationArray, validMask, metadata):
    transformTuple = buildTargetTransform(metadata)
    transformCoeffs = (transformTuple.a, transformTuple.b, transformTuple.c,
                       transformTuple.d, transformTuple.e, transformTuple.f)

    np.savez_compressed(
        outputPath,
        rgb=rgbArray,
        elevation=elevationArray,
        validMask=validMask,
        transform=transformCoeffs,
        crs='EPSG:32610',
        nodata=metadata.get('nodata_value', -9999)
    )

    print(f"\nSaved aligned data to {outputPath}")
    print(f"  RGB shape: {rgbArray.shape}")
    print(f"  Elevation shape: {elevationArray.shape}")
    print(f"  Valid pixels: {np.sum(validMask):,} ({100*np.sum(validMask)/validMask.size:.1f}%)")


def main():
    basePath = Path(__file__).parent
    elevPath = basePath / "Elevation" / "sfbaydeltadem10m2016.asc"
    tilesDir = basePath / "RGBImagery" / "tiles"
    outputPath = basePath / "aligned_data.npz"

    print("Loading elevation data...")
    elevation, elevMeta = loadElevation(elevPath)

    print("Building target coordinate system...")
    targetTransform = buildTargetTransform(elevMeta)
    targetCrs = CRS.from_epsg(32610)
    targetShape = (elevMeta['nrows'], elevMeta['ncols'])

    print(f"Target grid: {targetShape[0]} x {targetShape[1]} pixels at 10m resolution")

    print("\nPre-allocating RGB array...")
    rgbAligned = np.zeros((3, targetShape[0], targetShape[1]), dtype=np.uint8)

    print("\nReprojecting RGB tiles...")
    tilePaths = sorted(glob.glob(str(tilesDir / "bay_area_tile_*.tif")))
    print(f"Found {len(tilePaths)} tiles")

    rgbTemp = np.zeros_like(rgbAligned)

    for tilePath in tqdm(tilePaths, desc="Processing tiles"):
        # Clear temp buffer
        rgbTemp.fill(0)
        
        # Reproject to temp buffer
        reprojectTileToTarget(tilePath, rgbTemp, targetTransform, targetCrs)
        
        # Copy valid pixels to main array
        # Assuming 0 is nodata for RGB (black filler)
        validMaskTile = np.any(rgbTemp > 0, axis=0)
        rgbAligned[:, validMaskTile] = rgbTemp[:, validMaskTile]

    print("\nComputing intersection mask...")
    validMask = computeIntersectionMask(rgbAligned, elevation)

    print("Saving aligned data...")
    saveAlignedData(outputPath, rgbAligned, elevation, validMask, elevMeta)

    print("\n✓ Alignment complete!")


if __name__ == "__main__":
    main()
