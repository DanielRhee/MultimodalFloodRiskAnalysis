import torch
import numpy as np
from tqdm import tqdm

import config


class MetricsCalculator:
    def __init__(self, numClasses, ignoreClasses=None):
        self.numClasses = numClasses
        self.ignoreClasses = ignoreClasses if ignoreClasses else []
        self.reset()

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClasses, self.numClasses), dtype=np.int64)

    def update(self, preds, targets):
        preds = preds.cpu().numpy()
        targets = targets.cpu().numpy()

        mask = np.ones_like(targets, dtype=bool)
        for ignoreClass in self.ignoreClasses:
            mask &= (targets != ignoreClass)

        preds = preds[mask]
        targets = targets[mask]

        for pred, target in zip(preds.flatten(), targets.flatten()):
            if 0 <= pred < self.numClasses and 0 <= target < self.numClasses:
                self.confusionMatrix[target, pred] += 1

    def getIoU(self):
        iou = np.zeros(self.numClasses)
        for i in range(self.numClasses):
            if i in self.ignoreClasses:
                iou[i] = np.nan
                continue

            tp = self.confusionMatrix[i, i]
            fp = self.confusionMatrix[:, i].sum() - tp
            fn = self.confusionMatrix[i, :].sum() - tp

            denom = tp + fp + fn
            if denom > 0:
                iou[i] = tp / denom
            else:
                iou[i] = np.nan

        return iou

    def getMeanIoU(self):
        iou = self.getIoU()
        validIoU = iou[~np.isnan(iou)]
        if len(validIoU) > 0:
            return validIoU.mean()
        return 0.0

    def getOverallAccuracy(self):
        correctPredictions = np.diag(self.confusionMatrix).sum()
        totalPredictions = self.confusionMatrix.sum()
        if totalPredictions > 0:
            return correctPredictions / totalPredictions
        return 0.0

    def getMetrics(self):
        iou = self.getIoU()
        mIoU = self.getMeanIoU()
        oa = self.getOverallAccuracy()

        return {
            'mIoU': mIoU,
            'overallAccuracy': oa,
            'perClassIoU': iou,
            'confusionMatrix': self.confusionMatrix
        }


def evaluateModel(model, dataLoader, device, ignoreClasses=None):
    if ignoreClasses is None:
        ignoreClasses = config.IGNORED_CLASSES

    model.eval()
    metricsCalc = MetricsCalculator(config.NUM_CLASSES, ignoreClasses)

    with torch.no_grad():
        for images, labels in tqdm(dataLoader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            metricsCalc.update(preds, labels)

    return metricsCalc.getMetrics()


def printMetrics(metrics, classNames=None):
    if classNames is None:
        classNames = config.CLASS_NAMES

    print(f"\nOverall Metrics:")
    print(f"  mIoU: {metrics['mIoU']:.4f}")
    print(f"  Overall Accuracy: {metrics['overallAccuracy']:.4f}")

    print(f"\nPer-Class IoU:")
    for i, iou in enumerate(metrics['perClassIoU']):
        if not np.isnan(iou):
            className = classNames[i] if i < len(classNames) else f"Class {i}"
            print(f"  {className:25s}: {iou:.4f}")
