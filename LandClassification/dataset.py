import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio

import config


class FLAIRDataset(Dataset):
    def __init__(self, imagePaths, labelPaths, transform=None, isTrain=True):
        self.imagePaths = imagePaths
        self.labelPaths = labelPaths
        self.transform = transform
        self.isTrain = isTrain

        self.mean = torch.tensor(config.NORMALIZATION_MEAN).view(4, 1, 1)
        self.std = torch.tensor(config.NORMALIZATION_STD).view(4, 1, 1)

    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, idx):
        imagePath = self.imagePaths[idx]
        labelPath = self.labelPaths[idx]

        with rasterio.open(imagePath) as src:
            image = src.read()

        with rasterio.open(labelPath) as src:
            label = src.read(1)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()

        if self.transform:
            image, label = self.transform(image, label)

        image = (image - self.mean) / self.std

        return image, label


class RandomCrop:
    def __init__(self, cropSize):
        self.cropSize = cropSize

    def __call__(self, image, label):
        _, h, w = image.shape

        if h == self.cropSize and w == self.cropSize:
            return image, label

        top = random.randint(0, h - self.cropSize)
        left = random.randint(0, w - self.cropSize)

        image = image[:, top:top+self.cropSize, left:left+self.cropSize]
        label = label[top:top+self.cropSize, left:left+self.cropSize]

        return image, label


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = torch.flip(image, dims=[2])
            label = torch.flip(label, dims=[1])
        return image, label


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = torch.flip(image, dims=[1])
            label = torch.flip(label, dims=[0])
        return image, label


class RandomRotation90:
    def __call__(self, image, label):
        k = random.choice([0, 1, 2, 3])
        if k > 0:
            image = torch.rot90(image, k, dims=[1, 2])
            label = torch.rot90(label, k, dims=[0, 1])
        return image, label


class ComposeTransforms:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


def getImageLabelPairs(datasetRoot):
    imagePattern = os.path.join(datasetRoot, config.IMAGE_DIR_PATTERN, "*", "*.tif")
    imagePaths = sorted(glob.glob(imagePattern, recursive=True))

    labelPaths = []
    validImagePaths = []

    for imagePath in imagePaths:
        parts = imagePath.split(os.sep)
        filename = parts[-1]
        roi = parts[-2]
        departmentYear = parts[-3]

        department = departmentYear.split('_')[0]
        year = departmentYear.split('-')[1].split('_')[0]

        labelDirName = f"{department}-{year}_AERIAL_LABEL-COSIA"
        labelFilename = filename.replace("AERIAL_RGBI", "AERIAL_LABEL-COSIA")

        labelPath = os.path.join(datasetRoot, labelDirName, roi, labelFilename)

        if os.path.exists(labelPath):
            validImagePaths.append(imagePath)
            labelPaths.append(labelPath)

    return validImagePaths, labelPaths


def getDataLoaders(datasetRoot, batchSize=None, trainValSplit=None):
    if batchSize is None:
        batchSize = config.BATCH_SIZE
    if trainValSplit is None:
        trainValSplit = config.TRAIN_VAL_SPLIT

    imagePaths, labelPaths = getImageLabelPairs(datasetRoot)

    paired = list(zip(imagePaths, labelPaths))
    random.seed(config.RANDOM_SEED)
    random.shuffle(paired)
    imagePaths, labelPaths = zip(*paired)

    splitIdx = int(len(imagePaths) * trainValSplit)
    trainImagePaths = imagePaths[:splitIdx]
    trainLabelPaths = labelPaths[:splitIdx]
    valImagePaths = imagePaths[splitIdx:]
    valLabelPaths = labelPaths[splitIdx:]

    trainTransform = ComposeTransforms([
        RandomCrop(config.CROP_SIZE),
        RandomHorizontalFlip(0.5),
        RandomVerticalFlip(0.5),
        RandomRotation90()
    ])

    trainDataset = FLAIRDataset(
        trainImagePaths,
        trainLabelPaths,
        transform=trainTransform,
        isTrain=True
    )

    valDataset = FLAIRDataset(
        valImagePaths,
        valLabelPaths,
        transform=None,
        isTrain=False
    )

    trainLoader = DataLoader(
        trainDataset,
        batchSize=batchSize,
        shuffle=True,
        numWorkers=0,
        pinMemory=True
    )

    valLoader = DataLoader(
        valDataset,
        batchSize=1,
        shuffle=False,
        numWorkers=0,
        pinMemory=True
    )

    return trainLoader, valLoader, len(trainDataset), len(valDataset)
