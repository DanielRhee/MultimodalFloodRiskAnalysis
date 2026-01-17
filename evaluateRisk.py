import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys

import LandClassification.config as lc_config
from LandClassification.model import getMiniUNet
import RiskModel.config as rm_config
from RiskModel.model import getModel as getRiskModel

def crop_to_square(image):
    w, h = image.size
    size = min(w, h)
    return image.crop((0, 0, size, size))

class RiskEvaluator:
    def __init__(self, lc_checkpoint=None, rm_checkpoint=None, device=None):
        self.device = device or torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

        lc_checkpoint = lc_checkpoint or os.path.join('LandClassification', 'checkpoints', 'best_model.pth')
        self.lc_model = getMiniUNet(
            inChannels=lc_config.IN_CHANNELS,
            numClasses=lc_config.NUM_CLASSES,
            baseFilters=lc_config.BASE_FILTERS
        )
        lc_data = torch.load(lc_checkpoint, map_location=self.device, weights_only=False)
        self.lc_model.load_state_dict(lc_data['modelStateDict'])
        self.lc_model.to(self.device)
        self.lc_model.eval()

        rm_checkpoint = rm_checkpoint or os.path.join('RiskModel', 'checkpoints', 'best_model.pth')
        self.rm_model = getRiskModel(
            architecture='unet',
            inChannels=rm_config.INPUT_CHANNELS,
            outChannels=rm_config.OUTPUT_CLASSES,
            baseFilters=32
        )
        rm_data = torch.load(rm_checkpoint, map_location=self.device, weights_only=False)
        self.rm_model.load_state_dict(rm_data['modelStateDict'])
        self.rm_model.to(self.device)
        self.rm_model.eval()

    @torch.no_grad()
    def evaluate(self, image, tile_size=128, stride=128):
        w, h = image.size
        rgb_full = np.array(image).astype(np.float32)
        rm_rgb_norm = (rgb_full - np.array(rm_config.RGB_MEAN)) / np.array(rm_config.RGB_STD)
        dummy_elev_full = np.zeros((h, w, 1), dtype=np.float32)

        risk_mask = np.zeros((h, w), dtype=np.float32)
        count_mask = np.zeros((h, w), dtype=np.float32)

        print(f"Processing {h}x{w} image in {tile_size}x{tile_size} tiles...")

        for y in range(0, h - tile_size + 1, stride):
            for x in range(0, w - tile_size + 1, stride):
                rm_rgb_tile = rm_rgb_norm[y:y+tile_size, x:x+tile_size]
                elev_tile = dummy_elev_full[y:y+tile_size, x:x+tile_size]

                rm_input_np = np.concatenate([rm_rgb_tile, elev_tile], axis=2)
                rm_tensor = torch.from_numpy(rm_input_np.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)

                rm_output = self.rm_model(rm_tensor)
                risk_tile = rm_output.cpu().numpy()[0, 0]

                risk_mask[y:y+tile_size, x:x+tile_size] += risk_tile
                count_mask[y:y+tile_size, x:x+tile_size] += 1

        count_mask = np.maximum(count_mask, 1)
        risk_mask = (risk_mask / count_mask * 100).astype(np.uint8)

        return image, risk_mask

def visualize_risk(image, risk_mask, tile_size=128, output_path='risk_overlay.png'):
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # 1. Left View: Image with Grid
    axes[0].imshow(image)
    h, w = risk_mask.shape
    for y in range(0, h, tile_size):
        axes[0].axhline(y, color='white', lw=1, alpha=0.7)
    for x in range(0, w, tile_size):
        axes[0].axvline(x, color='white', lw=1, alpha=0.7)
    axes[0].set_title("Satellite Imagery with Analysis Grid", fontsize=15)
    axes[0].axis('off')

    # Map 0 values (no prediction) to nan for transparency or just use the cmap
    display_mask = risk_mask.astype(float)
    im = axes[1].imshow(display_mask, cmap='RdYlGn_r', vmin=0, vmax=100)
    for y in range(0, h + 1, tile_size):
        axes[1].axhline(y - 0.5, color='black', lw=0.5, alpha=0.3)
    for x in range(0, w + 1, tile_size):
        axes[1].axvline(x - 0.5, color='black', lw=0.5, alpha=0.3)

    axes[1].set_title("Predicted Flood Risk Heatmap (%)", fontsize=15)
    axes[1].axis('off')

    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Flood Risk Probability (%)', fontsize=12)

    plt.suptitle("Flood Risk Analysis Results", fontsize=20, y=0.95)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"Visualization saved to {output_path}")
    return fig

def main():
    print("Loading example data from validation set...")
    aligned_data_path = 'aligned_data.npz'
    if not os.path.exists(aligned_data_path):
        print(f"Error: {aligned_data_path} not found. please run preprocessing first.")
        return

    data = np.load(aligned_data_path)
    rgb_full = data['rgb']
    if rgb_full.shape[0] == 3:
        rgb_full = rgb_full.transpose(1, 2, 0)
    h_full, w_full, _ = rgb_full.shape
    size = 1024
    start_y, start_x = h_full//4, w_full//4
    sample_rgb = rgb_full[start_y:start_y+size, start_x:start_x+size]

    sample_pil = Image.fromarray(sample_rgb.astype(np.uint8))
    evaluator = RiskEvaluator()
    img, risk_mask = evaluator.evaluate(sample_pil, tile_size=128, stride=128)
    visualize_risk(img, risk_mask, tile_size=128)
    plt.show()

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
