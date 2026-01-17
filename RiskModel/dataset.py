import numpy as np
import torch
from torch.utils.data import Dataset
try:
    from . import config
except (ImportError, ValueError):
    import config

class FloodRiskDataset(Dataset):
    def __init__(self, alignedDataPath, floodRiskPath, split='train', tileSize=128, stride=64, minValidRatio=0.7, trainSplit=0.85):
        self.tileSize = tileSize
        self.stride = stride
        self.minValidRatio = minValidRatio

        print(f"Loading data for {split} split...")
        alignedData = np.load(alignedDataPath)
        floodData = np.load(floodRiskPath, allow_pickle=True)

        self.rgb = alignedData['rgb']
        if self.rgb.shape[0] == 3:
            self.rgb = np.transpose(self.rgb, (1, 2, 0))

        self.elevation = alignedData['elevation']
        self.validMask = alignedData['validMask']
        self.floodRisk = floodData['floodRisk']

        self.height, self.width = self.elevation.shape

        print(f"Data shape: {self.height} x {self.width}")
        print(f"Generating tile positions...")
        self.tiles = self._generateTiles()

        totalTiles = len(self.tiles)
        splitIdx = int(totalTiles * trainSplit)

        if split == 'train':
            self.tiles = self.tiles[:splitIdx]
        elif split == 'val':
            self.tiles = self.tiles[splitIdx:]
        else:
            raise ValueError(f"Invalid split: {split}")

        print(f"{split.capitalize()} tiles: {len(self.tiles)}")

    def _generateTiles(self):
        tiles = []

        for y in range(0, self.height - self.tileSize + 1, self.stride):
            for x in range(0, self.width - self.tileSize + 1, self.stride):
                validMaskTile = self.validMask[y:y+self.tileSize, x:x+self.tileSize]
                validRatio = validMaskTile.sum() / (self.tileSize * self.tileSize)

                if validRatio >= self.minValidRatio:
                    tiles.append((y, x))

        return tiles

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        y, x = self.tiles[idx]

        rgbTile = self.rgb[y:y+self.tileSize, x:x+self.tileSize].astype(np.float32)
        elevationTile = self.elevation[y:y+self.tileSize, x:x+self.tileSize].astype(np.float32)
        floodRiskTile = self.floodRisk[y:y+self.tileSize, x:x+self.tileSize].astype(np.float32)
        validMaskTile = self.validMask[y:y+self.tileSize, x:x+self.tileSize]

        rgbNorm = (rgbTile - np.array(config.RGB_MEAN)) / np.array(config.RGB_STD)

        elevMin = elevationTile[validMaskTile].min() if validMaskTile.any() else 0
        elevMax = elevationTile[validMaskTile].max() if validMaskTile.any() else 1
        elevRange = elevMax - elevMin
        if elevRange > 0:
            elevationNorm = (elevationTile - elevMin) / elevRange
        else:
            elevationNorm = np.zeros_like(elevationTile)

        inputTensor = np.concatenate([
            rgbNorm.transpose(2, 0, 1),
            elevationNorm[np.newaxis, :, :]
        ], axis=0)

        floodRiskNorm = floodRiskTile / 100.0

        # Handle NaNs and Infs
        if np.isnan(inputTensor).any() or np.isinf(inputTensor).any():
            inputTensor = np.nan_to_num(inputTensor, nan=0.0, posinf=0.0, neginf=0.0)
        
        if np.isnan(floodRiskNorm).any() or np.isinf(floodRiskNorm).any():
            # Update valid mask to exclude bad targets
            bad_targets = np.isnan(floodRiskNorm) | np.isinf(floodRiskNorm)
            validMaskTile = validMaskTile & ~bad_targets
            floodRiskNorm = np.nan_to_num(floodRiskNorm, nan=0.0, posinf=0.0, neginf=0.0)

        inputTensor = torch.from_numpy(inputTensor).float()
        targetTensor = torch.from_numpy(floodRiskNorm).float().unsqueeze(0)
        validMaskTensor = torch.from_numpy(validMaskTile).bool()

        return {
            'input': inputTensor,
            'target': targetTensor,
            'validMask': validMaskTensor,
            'position': (y, x)
        }

class FloodRiskInferenceDataset(Dataset):
    def __init__(self, alignedDataPath, tileSize=128, stride=64):
        self.tileSize = tileSize
        self.stride = stride

        print("Loading data for inference...")
        alignedData = np.load(alignedDataPath)

        self.rgb = alignedData['rgb']
        if self.rgb.shape[0] == 3:
            self.rgb = np.transpose(self.rgb, (1, 2, 0))

        self.elevation = alignedData['elevation']
        self.validMask = alignedData['validMask']

        self.height, self.width = self.elevation.shape

        print(f"Generating all tile positions...")
        self.tiles = self._generateAllTiles()
        print(f"Total tiles for inference: {len(self.tiles)}")

    def _generateAllTiles(self):
        tiles = []
        for y in range(0, self.height - self.tileSize + 1, self.stride):
            for x in range(0, self.width - self.tileSize + 1, self.stride):
                tiles.append((y, x))
        return tiles

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        y, x = self.tiles[idx]

        rgbTile = self.rgb[y:y+self.tileSize, x:x+self.tileSize].astype(np.float32)
        elevationTile = self.elevation[y:y+self.tileSize, x:x+self.tileSize].astype(np.float32)
        validMaskTile = self.validMask[y:y+self.tileSize, x:x+self.tileSize]

        rgbNorm = (rgbTile - np.array(config.RGB_MEAN)) / np.array(config.RGB_STD)

        elevMin = elevationTile[validMaskTile].min() if validMaskTile.any() else 0
        elevMax = elevationTile[validMaskTile].max() if validMaskTile.any() else 1
        elevRange = elevMax - elevMin
        if elevRange > 0:
            elevationNorm = (elevationTile - elevMin) / elevRange
        else:
            elevationNorm = np.zeros_like(elevationTile)

        inputTensor = np.concatenate([
            rgbNorm.transpose(2, 0, 1),
            elevationNorm[np.newaxis, :, :]
        ], axis=0)

        inputTensor = torch.from_numpy(inputTensor).float()
        validMaskTensor = torch.from_numpy(validMaskTile).bool()

        return {
            'input': inputTensor,
            'validMask': validMaskTensor,
            'position': (y, x)
        }
