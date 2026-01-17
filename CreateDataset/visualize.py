import numpy as np
import matplotlib.pyplot as plt

def visualizeFloodRisk():
    floodData = np.load('flood_risk_dataset.npz', allow_pickle=True)
    alignedData = np.load('../aligned_data.npz')

    floodRisk = floodData['floodRisk']
    validMask = floodData['validMask']
    metadata = floodData['metadata'].item()

    rgb = alignedData['rgb']
    if rgb.shape[0] == 3:
        rgb = np.transpose(rgb, (1, 2, 0))

    displayRisk = np.where(validMask, floodRisk, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    rgbDisplay = rgb.astype(np.float32) / 255.0
    rgbDisplay = np.clip(rgbDisplay * 2.0, 0, 1)
    axes[0].imshow(rgbDisplay)
    axes[0].set_title('RGB Imagery (SF Bay Area)', fontsize=14)
    axes[0].axis('off')

    im = axes[1].imshow(displayRisk, cmap='RdYlGn_r', vmin=0, vmax=100)
    axes[1].set_title('Flood Risk Score (0-100)', fontsize=14)
    axes[1].axis('off')

    cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Flood Risk', fontsize=12)

    plt.suptitle(f"Flood Risk Dataset - SF Bay Area\nCreated: {metadata['created']}", fontsize=16, y=0.98)
    plt.tight_layout()

    plt.savefig('flood_risk_visualization.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to flood_risk_visualization.png")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    validRisks = floodRisk[validMask]
    plt.hist(validRisks, bins=100, color='steelblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Flood Risk Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Flood Risk Scores', fontsize=14)
    plt.grid(axis='y', alpha=0.3)

    plt.subplot(1, 2, 2)
    bins = [0, 20, 40, 60, 80, 100]
    labels = ['Very Low\n(0-20)', 'Low\n(20-40)', 'Medium\n(40-60)', 'High\n(60-80)', 'Very High\n(80-100)']
    counts = []
    for i in range(len(bins)-1):
        count = np.sum((validRisks >= bins[i]) & (validRisks < bins[i+1]))
        counts.append(count)

    colors = ['green', 'yellowgreen', 'yellow', 'orange', 'red']
    plt.bar(labels, counts, color=colors, edgecolor='black', alpha=0.7)
    plt.ylabel('Number of Pixels', fontsize=12)
    plt.title('Flood Risk Categories', fontsize=14)
    plt.xticks(rotation=0, fontsize=10)
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('flood_risk_statistics.png', dpi=150, bbox_inches='tight')
    print("Saved statistics to flood_risk_statistics.png")

if __name__ == '__main__':
    visualizeFloodRisk()
