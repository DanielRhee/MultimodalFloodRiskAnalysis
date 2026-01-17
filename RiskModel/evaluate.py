import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

import config
from dataset import FloodRiskDataset
from model import getModel

def evaluateModel(modelPath, architecture='unet', baseFilters=32, batchSize=16):
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    print("\nLoading model...")
    model = getModel(
        architecture=architecture,
        inChannels=config.INPUT_CHANNELS,
        outChannels=config.OUTPUT_CLASSES,
        baseFilters=baseFilters
    )

    checkpoint = torch.load(modelPath, map_location=device)
    model.load_state_dict(checkpoint['modelStateDict'])
    model.to(device)
    model.eval()

    print("Loading validation dataset...")
    valDataset = FloodRiskDataset(
        config.ALIGNED_DATA_PATH,
        config.FLOOD_RISK_PATH,
        split='val',
        tileSize=config.TILE_SIZE,
        stride=config.STRIDE,
        minValidRatio=config.MIN_VALID_RATIO
    )

    valLoader = DataLoader(
        valDataset,
        batch_size=batchSize,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )

    print("\nEvaluating...")

    allPredictions = []
    allTargets = []
    allErrors = []

    with torch.no_grad():
        for batch in tqdm(valLoader):
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            validMask = batch['validMask'].to(device)

            outputs = model(inputs)

            validMaskExpanded = validMask.unsqueeze(1)
            validPreds = outputs[validMaskExpanded].cpu().numpy() * 100
            validTargets = targets[validMaskExpanded].cpu().numpy() * 100

            allPredictions.extend(validPreds)
            allTargets.extend(validTargets)
            allErrors.extend(np.abs(validPreds - validTargets))

    allPredictions = np.array(allPredictions)
    allTargets = np.array(allTargets)
    allErrors = np.array(allErrors)

    mae = allErrors.mean()
    rmse = np.sqrt((allErrors ** 2).mean())
    medianError = np.median(allErrors)
    percentile90 = np.percentile(allErrors, 90)
    percentile95 = np.percentile(allErrors, 95)

    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nError Metrics:")
    print(f"  MAE:              {mae:.2f}")
    print(f"  RMSE:             {rmse:.2f}")
    print(f"  Median Error:     {medianError:.2f}")
    print(f"  90th Percentile:  {percentile90:.2f}")
    print(f"  95th Percentile:  {percentile95:.2f}")

    print(f"\nPrediction Statistics:")
    print(f"  Mean:   {allPredictions.mean():.2f}")
    print(f"  Std:    {allPredictions.std():.2f}")
    print(f"  Min:    {allPredictions.min():.2f}")
    print(f"  Max:    {allPredictions.max():.2f}")

    print(f"\nGround Truth Statistics:")
    print(f"  Mean:   {allTargets.mean():.2f}")
    print(f"  Std:    {allTargets.std():.2f}")
    print(f"  Min:    {allTargets.min():.2f}")
    print(f"  Max:    {allTargets.max():.2f}")

    riskBins = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
    riskLabels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']

    print(f"\nError by Risk Category:")
    for (binMin, binMax), label in zip(riskBins, riskLabels):
        mask = (allTargets >= binMin) & (allTargets < binMax)
        if mask.sum() > 0:
            binMae = allErrors[mask].mean()
            binCount = mask.sum()
            print(f"  {label:12s} ({binMin:3d}-{binMax:3d}): MAE = {binMae:5.2f}  (n={binCount})")

    print("="*60)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelPath', type=str, required=True)
    parser.add_argument('--architecture', type=str, default='unet', choices=['unet', 'efficientnet'])
    parser.add_argument('--baseFilters', type=int, default=32)
    parser.add_argument('--batchSize', type=int, default=16)
    args = parser.parse_args()

    evaluateModel(
        modelPath=args.modelPath,
        architecture=args.architecture,
        baseFilters=args.baseFilters,
        batchSize=args.batchSize
    )

if __name__ == '__main__':
    main()
