import argparse
import os
import torch

import config
from dataset import getDataLoaders
from model import getMiniUNet, countParameters
from train import SegmentationTrainer, loadCheckpoint
from evaluate import evaluateModel, printMetrics
from utils import saveTrainingCurves, visualizeSamplePredictions, predictSingleImage, visualizePrediction


def train(args):
    print("=" * 60)
    print("FLAIR-Toy Semantic Segmentation Training")
    print("=" * 60)

    trainLoader, valLoader, numTrain, numVal = getDataLoaders(
        args.dataRoot,
        args.batchSize,
        args.trainValSplit
    )

    print(f"\nDataset loaded:")
    print(f"  Training samples: {numTrain}")
    print(f"  Validation samples: {numVal}")
    print(f"  Total samples: {numTrain + numVal}")

    model = getMiniUNet()
    numParams = countParameters(model)
    print(f"\nModel: Mini U-Net")
    print(f"  Parameters: {numParams:,}")

    trainer = SegmentationTrainer(model, trainLoader, valLoader, config.DEVICE)

    results = trainer.train(args.numEpochs)

    saveTrainingCurves(results['trainLosses'], results['valMIoUs'])

    print("\nGenerating sample predictions...")
    visualizeSamplePredictions(model, valLoader, config.DEVICE, numSamples=5)

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best mIoU: {results['bestMIoU']:.4f}")
    print("=" * 60)


def evaluate(args):
    print("=" * 60)
    print("FLAIR-Toy Semantic Segmentation Evaluation")
    print("=" * 60)

    _, valLoader, _, numVal = getDataLoaders(args.dataRoot)

    print(f"\nValidation samples: {numVal}")

    model = getMiniUNet()
    model, checkpoint = loadCheckpoint(model, args.checkpoint, config.DEVICE)

    print(f"\nLoaded checkpoint from epoch {checkpoint['epoch']}")

    metrics = evaluateModel(model, valLoader, config.DEVICE)

    printMetrics(metrics)

    print("\n" + "=" * 60)


def inference(args):
    print("=" * 60)
    print("FLAIR-Toy Semantic Segmentation Inference")
    print("=" * 60)

    model = getMiniUNet()
    model, checkpoint = loadCheckpoint(model, args.checkpoint, config.DEVICE)

    print(f"\nLoaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Processing image: {args.imagePath}")

    image, prediction = predictSingleImage(model, args.imagePath, config.DEVICE)

    outputPath = args.output if args.output else "prediction.png"
    visualizePrediction(image, prediction, prediction, savePath=outputPath)

    print(f"\nPrediction saved to {outputPath}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="FLAIR-Toy Semantic Segmentation")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    trainParser = subparsers.add_parser('train', help='Train the model')
    trainParser.add_argument('--dataRoot', type=str, default=config.DATASET_ROOT,
                            help='Path to dataset root')
    trainParser.add_argument('--batchSize', type=int, default=config.BATCH_SIZE,
                            help='Batch size for training')
    trainParser.add_argument('--numEpochs', type=int, default=config.NUM_EPOCHS,
                            help='Number of epochs')
    trainParser.add_argument('--trainValSplit', type=float, default=config.TRAIN_VAL_SPLIT,
                            help='Train/val split ratio')

    evalParser = subparsers.add_parser('evaluate', help='Evaluate the model')
    evalParser.add_argument('--dataRoot', type=str, default=config.DATASET_ROOT,
                           help='Path to dataset root')
    evalParser.add_argument('--checkpoint', type=str, required=True,
                           help='Path to model checkpoint')

    inferParser = subparsers.add_parser('inference', help='Run inference on a single image')
    inferParser.add_argument('--imagePath', type=str, required=True,
                            help='Path to input image')
    inferParser.add_argument('--checkpoint', type=str, required=True,
                            help='Path to model checkpoint')
    inferParser.add_argument('--output', type=str, default=None,
                            help='Path to save output')

    args = parser.parse_args()

    if args.command == 'train':
        train(args)
    elif args.command == 'evaluate':
        evaluate(args)
    elif args.command == 'inference':
        inference(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
