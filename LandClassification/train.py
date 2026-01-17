import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

import config
from evaluate import evaluateModel


class SegmentationTrainer:
    def __init__(self, model, trainLoader, valLoader, device):
        self.model = model.to(device)
        self.trainLoader = trainLoader
        self.valLoader = valLoader
        self.device = device

        classWeights = torch.ones(config.NUM_CLASSES)
        for ignoreClass in config.IGNORED_CLASSES:
            classWeights[ignoreClass] = 0.0
        classWeights = classWeights.to(device)

        self.criterion = nn.CrossEntropyLoss(weight=classWeights, ignore_index=-1)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.NUM_EPOCHS,
            eta_min=config.LR_MIN
        )

        self.bestMIoU = 0.0
        self.trainLosses = []
        self.valMIoUs = []

        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    def trainEpoch(self):
        self.model.train()
        totalLoss = 0.0
        numBatches = 0

        progressBar = tqdm(self.trainLoader, desc="Training")
        for images, labels in progressBar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            totalLoss += loss.item()
            numBatches += 1

            progressBar.set_postfix({'loss': f'{loss.item():.4f}'})

        avgLoss = totalLoss / numBatches
        return avgLoss

    def validate(self):
        metrics = evaluateModel(self.model, self.valLoader, self.device)
        return metrics

    def saveCheckpoint(self, epoch, metrics, isBest=False):
        checkpoint = {
            'epoch': epoch,
            'modelStateDict': self.model.state_dict(),
            'optimizerStateDict': self.optimizer.state_dict(),
            'schedulerStateDict': self.scheduler.state_dict(),
            'metrics': metrics,
            'bestMIoU': self.bestMIoU
        }

        checkpointPath = os.path.join(config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpointPath)

        if isBest:
            bestPath = os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
            torch.save(checkpoint, bestPath)

        if epoch % 5 != 0 and not isBest:
            if os.path.exists(checkpointPath):
                os.remove(checkpointPath)

    def train(self, numEpochs=None):
        if numEpochs is None:
            numEpochs = config.NUM_EPOCHS

        print(f"Starting training for {numEpochs} epochs")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.trainLoader.dataset)}")
        print(f"Validation samples: {len(self.valLoader.dataset)}")
        print(f"Batch size: {config.BATCH_SIZE}")

        for epoch in range(1, numEpochs + 1):
            print(f"\nEpoch {epoch}/{numEpochs}")
            print(f"Learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")

            trainLoss = self.trainEpoch()
            self.trainLosses.append(trainLoss)

            print(f"Train Loss: {trainLoss:.4f}")

            metrics = self.validate()
            mIoU = metrics['mIoU']
            oa = metrics['overallAccuracy']
            self.valMIoUs.append(mIoU)

            print(f"Validation mIoU: {mIoU:.4f}, Overall Accuracy: {oa:.4f}")

            isBest = mIoU > self.bestMIoU
            if isBest:
                self.bestMIoU = mIoU
                print(f"New best mIoU: {self.bestMIoU:.4f}")

            if epoch % 5 == 0 or isBest:
                self.saveCheckpoint(epoch, metrics, isBest)

            self.scheduler.step()

        print(f"\nTraining completed!")
        print(f"Best mIoU: {self.bestMIoU:.4f}")

        return {
            'trainLosses': self.trainLosses,
            'valMIoUs': self.valMIoUs,
            'bestMIoU': self.bestMIoU
        }


def loadCheckpoint(model, checkpointPath, device):
    checkpoint = torch.load(checkpointPath, map_location=device)
    model.load_state_dict(checkpoint['modelStateDict'])
    return model, checkpoint
