import numpy as np
from PIL import Image
import os
import json
import random

def extract_samples():
    # Configuration
    data_path = 'aligned_data.npz'
    output_dir = 'SampleInput'
    sample_count = 5
    tile_size = 512
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Load data
    print(f"Loading {data_path}...")
    try:
        data = np.load(data_path)
        # Based on alignData.py:
        # rgb shape: (3, H, W)
        # elevation shape: (H, W)
        # validMask shape: (H, W)
        rgb_data = data['rgb']
        elevation_data = data['elevation']
        valid_mask = data['validMask']
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    h, w = valid_mask.shape
    print(f"Data shape: {h}x{w}")
    
    samples_collected = 0
    attempts = 0
    max_attempts = 1000
    
    while samples_collected < sample_count and attempts < max_attempts:
        attempts += 1
        
        # Pick random top-left corner
        y = random.randint(0, h - tile_size)
        x = random.randint(0, w - tile_size)
        
        # Check if crop is fully valid
        crop_mask = valid_mask[y:y+tile_size, x:x+tile_size]
        if not np.all(crop_mask):
            continue
            
        print(f"Found valid sample {samples_collected+1} at y={y}, x={x}")
        
        # Extract RGB
        # Transpose from (3, H, W) to (H, W, 3) for PIL
        rgb_crop = rgb_data[:, y:y+tile_size, x:x+tile_size]
        rgb_img_array = np.transpose(rgb_crop, (1, 2, 0))
        
        # Extract Elevation
        elev_crop = elevation_data[y:y+tile_size, x:x+tile_size]
        
        # Calculate min/max for normalization
        depth_min = float(np.min(elev_crop))
        depth_max = float(np.max(elev_crop))
        
        # Avoid division by zero if flat
        if depth_max == depth_min:
            normalized_elev = np.zeros_like(elev_crop, dtype=np.uint8)
        else:
            normalized_elev = ((elev_crop - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
            
        # Save RGB Image
        rgb_img = Image.fromarray(rgb_img_array)
        rgb_filename = f"sample_{samples_collected}_rgb.png"
        rgb_img.save(os.path.join(output_dir, rgb_filename))
        
        # Save Depth Image (Grayscale)
        depth_img = Image.fromarray(normalized_elev, mode='L')
        depth_filename = f"sample_{samples_collected}_depth.png"
        depth_img.save(os.path.join(output_dir, depth_filename))
        
        # Save Metadata
        info = {
            "depthMin": depth_min,
            "depthMax": depth_max,
            "original_x": x,
            "original_y": y
        }
        info_filename = f"sample_{samples_collected}_info.json"
        with open(os.path.join(output_dir, info_filename), 'w') as f:
            json.dump(info, f, indent=2)
            
        samples_collected += 1

    if samples_collected < sample_count:
        print(f"Warning: Could only find {samples_collected} valid samples after {attempts} attempts.")
    else:
        print(f"Successfully extracted {samples_collected} samples to {output_dir}/")

if __name__ == "__main__":
    extract_samples()
