import numpy as np
import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'LandClassification'))

import config

def loadLandCoverModel():
    try:
        from model import getMiniUNet

        modelPath = '../LandClassification/checkpoints/best_model.pth'
        if not os.path.exists(modelPath):
            print("Model not found, using fallback RGB-based features")
            return None

        device = torch.device('cpu')
        model = getMiniUNet(inChannels=4, numClasses=19, baseFilters=32)
        checkpoint = torch.load(modelPath, map_location=device, weights_only=False)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()
        print("Loaded land cover classification model")
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Using fallback RGB-based features")
        return None

def estimateInfrared(rgb):
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)

    nir = np.clip(1.5 * r - 0.5 * g, 0, 255)
    return nir.astype(np.uint8)

def classifyWithModel(model, rgb, validMask):
    device = torch.device('cpu')
    height, width = rgb.shape[:2]

    nir = estimateInfrared(rgb)
    rgbi = np.dstack([rgb, nir[:, :, None]])

    mean = np.array([104.0, 114.0, 99.0, 112.0])
    std = np.array([52.0, 46.0, 44.0, 49.0])

    rgbiNorm = (rgbi.astype(np.float32) - mean) / std
    rgbiNorm = rgbiNorm.transpose(2, 0, 1)
    rgbiTensor = torch.from_numpy(rgbiNorm).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(rgbiTensor)
        predictions = output.argmax(dim=1).squeeze().cpu().numpy()

    return predictions

def fallbackRGBClassification(rgb, validMask):
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)

    classifications = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)

    ndvi = (r - g) / (r + g + 1e-6)
    ndwi = (g - r) / (g + r + 1e-6)
    brightness = (r + g + b) / 3

    waterMask = (ndwi > 0.3) & (b > 100)
    classifications[waterMask] = 6

    imperviousMask = (brightness > 150) & (np.abs(r - g) < 20) & (np.abs(g - b) < 20) & ~waterMask
    classifications[imperviousMask] = 3

    vegetationMask = (ndvi > 0.2) & (g > r) & ~waterMask & ~imperviousMask
    classifications[vegetationMask] = 8

    bareSoilMask = (brightness > 100) & (r > g) & (g > b) & ~waterMask & ~imperviousMask & ~vegetationMask
    classifications[bareSoilMask] = 5

    return classifications

def calculateLandCoverRisk(rgb, validMask, model=None):
    print("Calculating land cover-based flood risk...")

    if model is not None:
        classifications = classifyWithModel(model, rgb, validMask)
    else:
        classifications = fallbackRGBClassification(rgb, validMask)

    landCoverRisk = np.zeros_like(classifications, dtype=np.float32)

    for classId, risk in config.LANDCOVER_RISK.items():
        landCoverRisk[classifications == classId] = risk

    landCoverRisk[~validMask] = 0

    uniqueClasses, counts = np.unique(classifications[validMask], return_counts=True)
    print(f"Land cover classes found: {len(uniqueClasses)}")
    for classId, count in zip(uniqueClasses, counts):
        risk = config.LANDCOVER_RISK.get(classId, 0.2)
        pct = 100 * count / validMask.sum()
        print(f"  Class {classId}: {count:,} pixels ({pct:.2f}%), risk={risk}")

    print(f"Land cover risk range: [{landCoverRisk[validMask].min():.4f}, {landCoverRisk[validMask].max():.4f}]")

    return landCoverRisk
