import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import rasterio

import config


def visualizePrediction(image, label, prediction, classColors=None, savePath=None):
    if classColors is None:
        classColors = config.CLASS_COLORS

    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(label, torch.Tensor):
        label = label.cpu().numpy()
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().numpy()

    if image.shape[0] == 4:
        imageRGB = image[:3]
        imageRGB = np.transpose(imageRGB, (1, 2, 0))
        imageRGB = (imageRGB - imageRGB.min()) / (imageRGB.max() - imageRGB.min() + 1e-8)
    else:
        imageRGB = image

    labelColor = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for classIdx, color in enumerate(classColors):
        mask = (label == classIdx)
        labelColor[mask] = color

    predColor = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
    for classIdx, color in enumerate(classColors):
        mask = (prediction == classIdx)
        predColor[mask] = color

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(imageRGB)
    axes[0].set_title('Input Image (RGB)')
    axes[0].axis('off')

    axes[1].imshow(labelColor)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    axes[2].imshow(predColor)
    axes[2].set_title('Prediction')
    axes[2].axis('off')

    plt.tight_layout()

    if savePath:
        os.makedirs(os.path.dirname(savePath), exist_ok=True)
        plt.savefig(savePath, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def predictSingleImage(model, imagePath, device):
    model.eval()

    with rasterio.open(imagePath) as src:
        image = src.read()

    imageTensor = torch.from_numpy(image).float().unsqueeze(0)

    mean = torch.tensor(config.NORMALIZATION_MEAN).view(1, 4, 1, 1)
    std = torch.tensor(config.NORMALIZATION_STD).view(1, 4, 1, 1)
    imageTensor = (imageTensor - mean) / std

    imageTensor = imageTensor.to(device)

    with torch.no_grad():
        output = model(imageTensor)
        prediction = output.argmax(dim=1).squeeze(0)

    return image, prediction


def saveTrainingCurves(trainLosses, valMIoUs, savePath=None):
    if savePath is None:
        savePath = os.path.join(config.RESULTS_DIR, 'training_curves.png')

    os.makedirs(os.path.dirname(savePath), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(trainLosses)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True)

    axes[1].plot(valMIoUs)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('mIoU')
    axes[1].set_title('Validation mIoU')
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(savePath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Training curves saved to {savePath}")


def visualizeSamplePredictions(model, dataLoader, device, numSamples=5, savePath=None):
    if savePath is None:
        savePath = config.RESULTS_DIR

    os.makedirs(savePath, exist_ok=True)

    model.eval()
    sampleCount = 0

    with torch.no_grad():
        for images, labels in dataLoader:
            if sampleCount >= numSamples:
                break

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            for i in range(images.size(0)):
                if sampleCount >= numSamples:
                    break

                image = images[i]
                label = labels[i]
                prediction = predictions[i]

                sampleSavePath = os.path.join(savePath, f'sample_{sampleCount}.png')
                visualizePrediction(image, label, prediction, savePath=sampleSavePath)

                sampleCount += 1

    print(f"Sample predictions saved to {savePath}")
