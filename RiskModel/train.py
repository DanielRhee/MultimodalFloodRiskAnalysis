import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
import argparse

try:
    from . import config
    from .dataset import FloodRiskDataset
    from .model import getModel
except (ImportError, ValueError):
    import config
    from dataset import FloodRiskDataset
    from model import getModel

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets, validMask):
        validMask = validMask.unsqueeze(1).float()
        loss = (predictions - targets) ** 2
        maskedLoss = loss * validMask
        return maskedLoss.sum() / (validMask.sum() + 1e-6)

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets, validMask):
        validMask = validMask.unsqueeze(1).float()
        loss = torch.abs(predictions - targets)
        maskedLoss = loss * validMask
        return maskedLoss.sum() / (validMask.sum() + 1e-6)

class CombinedLoss(nn.Module):
    def __init__(self, mseWeight=0.7, l1Weight=0.3):
        super().__init__()
        self.mseWeight = mseWeight
        self.l1Weight = l1Weight
        self.mseLoss = MaskedMSELoss()
        self.l1Loss = MaskedL1Loss()

    def forward(self, predictions, targets, validMask):
        mseLoss = self.mseLoss(predictions, targets, validMask)
        l1Loss = self.l1Loss(predictions, targets, validMask)
        return self.mseWeight * mseLoss + self.l1Weight * l1Loss

def calculateMetrics(predictions, targets, validMask):
    validMask = validMask.unsqueeze(1)
    validPreds = predictions[validMask].cpu().numpy()
    validTargets = targets[validMask].cpu().numpy()

    mae = np.abs(validPreds - validTargets).mean() * 100
    rmse = np.sqrt(((validPreds - validTargets) ** 2).mean()) * 100

    return {'mae': mae, 'rmse': rmse}

def trainEpoch(model, dataloader, criterion, optimizer, device):
    model.train()
    totalLoss = 0.0
    totalMae = 0.0
    totalRmse = 0.0
    numBatches = 0

    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        inputs = batch['input'].to(device)
        targets = batch['target'].to(device)
        validMask = batch['validMask'].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets, validMask)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        totalLoss += loss.item()

        with torch.no_grad():
            metrics = calculateMetrics(outputs, targets, validMask)
            totalMae += metrics['mae']
            totalRmse += metrics['rmse']

        numBatches += 1
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'mae': f"{metrics['mae']:.2f}"})

    return {
        'loss': totalLoss / numBatches,
        'mae': totalMae / numBatches,
        'rmse': totalRmse / numBatches
    }

def validateEpoch(model, dataloader, criterion, device):
    model.eval()
    totalLoss = 0.0
    totalMae = 0.0
    totalRmse = 0.0
    numBatches = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for batch in pbar:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            validMask = batch['validMask'].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets, validMask)

            totalLoss += loss.item()

            metrics = calculateMetrics(outputs, targets, validMask)
            totalMae += metrics['mae']
            totalRmse += metrics['rmse']

            numBatches += 1
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'mae': f"{metrics['mae']:.2f}"})

    return {
        'loss': totalLoss / numBatches,
        'mae': totalMae / numBatches,
        'rmse': totalRmse / numBatches
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', type=str, default='unet', choices=['unet', 'efficientnet'])
    parser.add_argument('--baseFilters', type=int, default=32)
    parser.add_argument('--batchSize', type=int, default=config.BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=config.EPOCHS)
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")
    
    # Enable anomaly detection for debugging
    torch.autograd.set_detect_anomaly(True)

    print("\nCreating datasets...")
    trainDataset = FloodRiskDataset(
        config.ALIGNED_DATA_PATH,
        config.FLOOD_RISK_PATH,
        split='train',
        tileSize=config.TILE_SIZE,
        stride=config.STRIDE,
        minValidRatio=config.MIN_VALID_RATIO
    )

    valDataset = FloodRiskDataset(
        config.ALIGNED_DATA_PATH,
        config.FLOOD_RISK_PATH,
        split='val',
        tileSize=config.TILE_SIZE,
        stride=config.STRIDE,
        minValidRatio=config.MIN_VALID_RATIO
    )

    trainLoader = DataLoader(
        trainDataset,
        batch_size=args.batchSize,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=False
    )

    valLoader = DataLoader(
        valDataset,
        batch_size=args.batchSize,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=False
    )

    print("\nCreating model...")
    model = getModel(
        architecture=args.architecture,
        inChannels=config.INPUT_CHANNELS,
        outChannels=config.OUTPUT_CLASSES,
        baseFilters=args.baseFilters
    )
    model = model.to(device)

    totalParams = sum(p.numel() for p in model.parameters())
    trainableParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {totalParams:,}")
    print(f"Trainable parameters: {trainableParams:,}")

    criterion = CombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    startEpoch = 0
    bestValLoss = float('inf')

    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['modelStateDict'])
        optimizer.load_state_dict(checkpoint['optimizerStateDict'])
        startEpoch = checkpoint['epoch'] + 1
        bestValLoss = checkpoint['bestValLoss']

    print(f"\nStarting training from epoch {startEpoch}...")

    for epoch in range(startEpoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 60)

        trainMetrics = trainEpoch(model, trainLoader, criterion, optimizer, device)
        valMetrics = validateEpoch(model, valLoader, criterion, device)

        scheduler.step(valMetrics['loss'])

        print(f"\nTrain - Loss: {trainMetrics['loss']:.4f}, MAE: {trainMetrics['mae']:.2f}, RMSE: {trainMetrics['rmse']:.2f}")
        print(f"Val   - Loss: {valMetrics['loss']:.4f}, MAE: {valMetrics['mae']:.2f}, RMSE: {valMetrics['rmse']:.2f}")

        checkpointPath = os.path.join(config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch,
            'modelStateDict': model.state_dict(),
            'optimizerStateDict': optimizer.state_dict(),
            'trainLoss': trainMetrics['loss'],
            'valLoss': valMetrics['loss'],
            'bestValLoss': bestValLoss,
        }, checkpointPath)

        if valMetrics['loss'] < bestValLoss:
            bestValLoss = valMetrics['loss']
            bestModelPath = os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'modelStateDict': model.state_dict(),
                'optimizerStateDict': optimizer.state_dict(),
                'valLoss': valMetrics['loss'],
                'valMae': valMetrics['mae'],
                'valRmse': valMetrics['rmse'],
            }, bestModelPath)
            print(f"Saved best model with val loss: {bestValLoss:.4f}")

    print("\nTraining completed!")
    print(f"Best validation loss: {bestValLoss:.4f}")

if __name__ == '__main__':
    main()
