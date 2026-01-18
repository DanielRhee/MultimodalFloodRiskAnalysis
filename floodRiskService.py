import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO

from evaluateRiskModel import RiskEvaluator
import LandClassification.config as lc_config

class FloodRiskService:
    def __init__(self):
        self.evaluator = RiskEvaluator()

    def analyze(self, image, depthMap=None, tileSize=128, stride=128):
        _, riskMask = self.evaluator.evaluate(image, depthMap=depthMap, tile_size=tileSize, stride=stride)
        landMask = self.evaluator.getLandClassification(image)

        stats = self.computeStatistics(riskMask, landMask)

        return {
            'riskMask': riskMask,
            'landMask': landMask,
            'averageRisk': stats['averageRisk'],
            'riskByLandClass': stats['riskByLandClass'],
            'landClassDistribution': stats['landClassDistribution'],
            'imageWidth': image.size[0],
            'imageHeight': image.size[1]
        }

    def generateVisualization(self, image, riskMask, landMask):
        fig, axes = plt.subplots(1, 3, figsize=(30, 10))

        axes[0].imshow(image)
        axes[0].set_title("Original Image", fontsize=15)
        axes[0].axis('off')

        riskDisplay = riskMask.astype(float)
        im1 = axes[1].imshow(riskDisplay, cmap='RdYlGn_r', vmin=0, vmax=100)
        axes[1].set_title("Flood Risk Heatmap (%)", fontsize=15)
        axes[1].axis('off')
        cbar1 = fig.colorbar(im1, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04)
        cbar1.set_label('Risk Probability (%)', fontsize=12)

        h, w = landMask.shape
        landColorMap = np.zeros((h, w, 3), dtype=np.uint8)
        for classIdx in range(lc_config.NUM_CLASSES):
            mask = landMask == classIdx
            landColorMap[mask] = lc_config.CLASS_COLORS[classIdx]

        axes[2].imshow(landColorMap)
        axes[2].set_title("Land Classification", fontsize=15)
        axes[2].axis('off')

        plt.suptitle("Flood Risk Analysis Results", fontsize=20, y=0.95)

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

    def generateRiskMapBytes(self, riskMask):
        fig, ax = plt.subplots(figsize=(10, 10))
        riskDisplay = riskMask.astype(float)
        im = ax.imshow(riskDisplay, cmap='RdYlGn_r', vmin=0, vmax=100)
        ax.set_title("Flood Risk Heatmap (%)", fontsize=15)
        ax.axis('off')
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('Risk Probability (%)', fontsize=12)

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

    def generateLandClassificationBytes(self, landMask):
        h, w = landMask.shape
        landColorMap = np.zeros((h, w, 3), dtype=np.uint8)
        for classIdx in range(lc_config.NUM_CLASSES):
            mask = landMask == classIdx
            landColorMap[mask] = lc_config.CLASS_COLORS[classIdx]

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(landColorMap)
        ax.set_title("Land Classification", fontsize=15)
        ax.axis('off')

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

    def computeStatistics(self, riskMask, landMask):
        averageRisk = float(np.mean(riskMask))

        riskByLandClass = {}
        landClassDistribution = {}

        totalPixels = landMask.size

        for classIdx in range(lc_config.NUM_CLASSES):
            mask = landMask == classIdx
            pixelCount = np.sum(mask)

            if pixelCount > 0:
                className = lc_config.CLASS_NAMES[classIdx]
                classRisk = riskMask[mask]
                riskByLandClass[className] = float(np.mean(classRisk))
                landClassDistribution[className] = float(pixelCount / totalPixels * 100)

        return {
            'averageRisk': averageRisk,
            'riskByLandClass': riskByLandClass,
            'landClassDistribution': landClassDistribution
        }
