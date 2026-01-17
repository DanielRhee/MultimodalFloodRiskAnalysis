import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os
import matplotlib.pyplot as plt

import config
from dataset import FloodRiskInferenceDataset
from model import getModel

class FloodRiskPredictor:
    def __init__(self, modelPath, architecture='unet', baseFilters=32, device=None):
        self.device = device or torch.device(config.DEVICE)

        print(f"Loading model from {modelPath}...")
        self.model = getModel(
            architecture=architecture,
            inChannels=config.INPUT_CHANNELS,
            outChannels=config.OUTPUT_CLASSES,
            baseFilters=baseFilters
        )

        checkpoint = torch.load(modelPath, map_location=self.device)
        self.model.load_state_dict(checkpoint['modelStateDict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded successfully (device: {self.device})")

    @torch.no_grad()
    def predictFullImage(self, alignedDataPath, tileSize=128, stride=64, batchSize=16):
        print("\nGenerating predictions for full image...")

        inferenceDataset = FloodRiskInferenceDataset(
            alignedDataPath,
            tileSize=tileSize,
            stride=stride
        )

        dataloader = DataLoader(
            inferenceDataset,
            batch_size=batchSize,
            shuffle=False,
            num_workers=config.NUM_WORKERS
        )

        height = inferenceDataset.height
        width = inferenceDataset.width

        predictionMap = np.zeros((height, width), dtype=np.float32)
        countMap = np.zeros((height, width), dtype=np.float32)

        print("Running inference...")
        for batch in tqdm(dataloader):
            inputs = batch['input'].to(self.device)
            validMask = batch['validMask']
            positions = batch['position']

            outputs = self.model(inputs)
            predictions = outputs.cpu().numpy()

            for i in range(predictions.shape[0]):
                y, x = positions[0][i].item(), positions[1][i].item()
                predTile = predictions[i, 0]
                maskTile = validMask[i].numpy()

                predictionMap[y:y+tileSize, x:x+tileSize] += predTile * maskTile
                countMap[y:y+tileSize, x:x+tileSize] += maskTile

        countMap = np.maximum(countMap, 1)
        predictionMap = predictionMap / countMap

        predictionMap = (predictionMap * 100).astype(np.uint8)

        return predictionMap

    @torch.no_grad()
    def predictTile(self, rgbTile, elevationTile):
        rgbNorm = (rgbTile - np.array(config.RGB_MEAN)) / np.array(config.RGB_STD)

        elevMin = elevationTile.min()
        elevMax = elevationTile.max()
        elevRange = elevMax - elevMin
        if elevRange > 0:
            elevationNorm = (elevationTile - elevMin) / elevRange
        else:
            elevationNorm = np.zeros_like(elevationTile)

        inputTensor = np.concatenate([
            rgbNorm.transpose(2, 0, 1),
            elevationNorm[np.newaxis, :, :]
        ], axis=0)

        inputTensor = torch.from_numpy(inputTensor).float().unsqueeze(0).to(self.device)

        output = self.model(inputTensor)
        prediction = output.cpu().numpy()[0, 0]

        return (prediction * 100).astype(np.uint8)

def visualizePredictions(predictions, validMask, groundTruth=None, outputPath='prediction_viz.png'):
    if groundTruth is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        displayPred = np.where(validMask, predictions, np.nan)
        displayGt = np.where(validMask, groundTruth, np.nan)
        displayDiff = np.where(validMask, np.abs(predictions - groundTruth), np.nan)

        im1 = axes[0].imshow(displayPred, cmap='RdYlGn_r', vmin=0, vmax=100)
        axes[0].set_title('Predicted Flood Risk', fontsize=14)
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046)

        im2 = axes[1].imshow(displayGt, cmap='RdYlGn_r', vmin=0, vmax=100)
        axes[1].set_title('Ground Truth Flood Risk', fontsize=14)
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046)

        im3 = axes[2].imshow(displayDiff, cmap='hot', vmin=0, vmax=30)
        axes[2].set_title('Absolute Error', fontsize=14)
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], fraction=0.046)

        validDiff = displayDiff[validMask]
        mae = np.nanmean(validDiff)
        rmse = np.sqrt(np.nanmean(validDiff ** 2))

        plt.suptitle(f'Flood Risk Prediction Results\nMAE: {mae:.2f}, RMSE: {rmse:.2f}', fontsize=16)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        displayPred = np.where(validMask, predictions, np.nan)

        im = ax.imshow(displayPred, cmap='RdYlGn_r', vmin=0, vmax=100)
        ax.set_title('Predicted Flood Risk', fontsize=14)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

        plt.suptitle('Flood Risk Prediction', fontsize=16)

    plt.tight_layout()
    plt.savefig(outputPath, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {outputPath}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelPath', type=str, required=True)
    parser.add_argument('--architecture', type=str, default='unet', choices=['unet', 'efficientnet'])
    parser.add_argument('--baseFilters', type=int, default=32)
    parser.add_argument('--batchSize', type=int, default=16)
    parser.add_argument('--outputDir', type=str, default='predictions')
    parser.add_argument('--compareGroundTruth', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.outputDir, exist_ok=True)

    predictor = FloodRiskPredictor(
        modelPath=args.modelPath,
        architecture=args.architecture,
        baseFilters=args.baseFilters
    )

    predictions = predictor.predictFullImage(
        alignedDataPath=config.ALIGNED_DATA_PATH,
        tileSize=config.TILE_SIZE,
        stride=config.STRIDE,
        batchSize=args.batchSize
    )

    alignedData = np.load(config.ALIGNED_DATA_PATH)
    validMask = alignedData['validMask']

    outputPath = os.path.join(args.outputDir, 'predictions.npz')
    np.savez_compressed(
        outputPath,
        predictions=predictions,
        validMask=validMask
    )
    print(f"\nPredictions saved to {outputPath}")

    groundTruth = None
    if args.compareGroundTruth:
        floodData = np.load(config.FLOOD_RISK_PATH)
        groundTruth = floodData['floodRisk']

    vizPath = os.path.join(args.outputDir, 'prediction_visualization.png')
    visualizePredictions(predictions, validMask, groundTruth, vizPath)

    validPredictions = predictions[validMask]
    print(f"\nPrediction Statistics:")
    print(f"  Min: {validPredictions.min()}")
    print(f"  Max: {validPredictions.max()}")
    print(f"  Mean: {validPredictions.mean():.2f}")
    print(f"  Median: {np.median(validPredictions):.2f}")
    print(f"  Std: {validPredictions.std():.2f}")

    if groundTruth is not None:
        validGt = groundTruth[validMask]
        mae = np.abs(validPredictions - validGt).mean()
        rmse = np.sqrt(((validPredictions - validGt) ** 2).mean())

        print(f"\nComparison with Ground Truth:")
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")

    print("\nInference completed!")

if __name__ == '__main__':
    main()
