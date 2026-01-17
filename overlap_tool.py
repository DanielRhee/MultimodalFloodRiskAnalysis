#!/usr/bin/env python3
"""
Advanced Overlap Extraction Tool

Displays RGB imagery and elevation data with overview and zoom panels,
allowing precise manual alignment and extraction.

Usage:
    python overlap_tool.py

Controls:
    - Click and drag on OVERVIEW or ZOOM panels to select a region
    - Use sliders or type in text boxes to adjust alignment parameters
    - Click "Export" button to save cropped regions
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Button, Slider, TextBox
from PIL import Image
from pathlib import Path

# Allow loading very large images
Image.MAX_IMAGE_PIXELS = None

# Target preview size (pixels) - keeps display responsive
PREVIEW_MAX_DIM = 2000


class OverlapExtractionTool:
    """Interactive tool for selecting and extracting overlapping regions."""
    
    def __init__(self, rgb_path: str, elevation_path: str, output_dir: str = "output"):
        self.rgb_path = Path(rgb_path)
        self.elevation_path = Path(elevation_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Selection state (in full-res RGB coordinates)
        self.selection = None  # (fx1, fy1, fx2, fy2)
        self.preview_selection = None # (px1, py1, px2, py2)
        
        # Alignment parameters
        self.params = {
            'scale': 1.0,
            'x_off': 0,
            'y_off': 0
        }
        
        # Load data (downsampled for preview)
        self._load_rgb()
        self._load_elevation()
        
        # Create UI
        self._setup_ui()
    
    def _load_rgb(self):
        """Load RGB imagery with downsampled preview."""
        print(f"Loading RGB imagery from {self.rgb_path}...")
        self.rgb_image = Image.open(self.rgb_path)
        self.rgb_full_size = self.rgb_image.size
        
        max_dim = max(self.rgb_full_size)
        self.rgb_scale = min(1.0, PREVIEW_MAX_DIM / max_dim)
        
        if self.rgb_scale < 1.0:
            preview_size = (
                int(self.rgb_full_size[0] * self.rgb_scale),
                int(self.rgb_full_size[1] * self.rgb_scale)
            )
            print(f"  Creating preview at {preview_size}...")
            rgb_preview = self.rgb_image.resize(preview_size, Image.Resampling.LANCZOS)
            self.rgb_array = np.array(rgb_preview)
        else:
            self.rgb_array = np.array(self.rgb_image)
    
    def _load_elevation(self):
        """Load elevation data from ASC file with downsampled preview."""
        print(f"Loading elevation data from {self.elevation_path}...")
        
        header = {}
        with open(self.elevation_path, 'r') as f:
            for _ in range(6):
                line = f.readline().strip()
                key, value = line.split()
                header[key.lower()] = float(value) if '.' in value else int(value)
        
        self.elev_header = header
        print("  Loading elevation values...")
        self.elevation_data = np.loadtxt(self.elevation_path, skiprows=6, dtype=np.float32)
        nodata = header.get('nodata_value', -9999)
        self.elevation_data[self.elevation_data == nodata] = np.nan
        
        max_dim = max(self.elevation_data.shape)
        self.elev_scale = min(1.0, PREVIEW_MAX_DIM / max_dim)
        
        if self.elev_scale < 1.0:
            from scipy.ndimage import zoom
            print(f"  Creating elevation preview at {self.elev_scale:.2%} scale...")
            self.elevation_preview = zoom(self.elevation_data, self.elev_scale, order=1)
        else:
            self.elevation_preview = self.elevation_data

    def _setup_ui(self):
        """Set up the 4-panel UI with side-by-side overview and zoom."""
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('Overlap Extraction Tool - Manual Alignment', fontsize=14)
        
        # Layout: 2x2 grid for images
        # [RGB Overview] [Elev Overview]
        # [RGB Zoom]     [Elev Zoom]
        gs = self.fig.add_gridspec(2, 2, height_ratios=[1, 1], left=0.05, right=0.95, top=0.92, bottom=0.25, hspace=0.2, wspace=0.15)
        
        self.ax_rgb_ov = self.fig.add_subplot(gs[0, 0])
        self.ax_elev_ov = self.fig.add_subplot(gs[0, 1])
        self.ax_rgb_zoom = self.fig.add_subplot(gs[1, 0])
        self.ax_elev_zoom = self.fig.add_subplot(gs[1, 1])
        
        # Initial displays
        self.ax_rgb_ov.set_title('RGB Overview')
        self.ax_rgb_ov.imshow(self.rgb_array)
        
        self.ax_elev_ov.set_title('Elevation Overview')
        self.elev_ov_img = self.ax_elev_ov.imshow(self.elevation_preview, cmap='terrain')
        
        self.ax_rgb_zoom.set_title('RGB Zoom Area')
        self.rgb_zoom_img = self.ax_rgb_zoom.imshow(self.rgb_array)
        
        self.ax_elev_zoom.set_title('Elevation Zoom Area')
        self.elev_zoom_img = self.ax_elev_zoom.imshow(self.elevation_preview, cmap='terrain')
        
        # Selection rectangles for overviews
        from matplotlib.patches import Rectangle
        self.rect_rgb_ov = Rectangle((0,0), 0, 0, linewidth=1, edgecolor='red', facecolor='none')
        self.rect_elev_ov = Rectangle((0,0), 0, 0, linewidth=1, edgecolor='yellow', facecolor='none')
        self.ax_rgb_ov.add_patch(self.rect_rgb_ov)
        self.ax_elev_ov.add_patch(self.rect_elev_ov)
        
        # Colorbar
        cbar_ax = self.fig.add_axes([0.96, 0.25, 0.01, 0.67])
        self.fig.colorbar(self.elev_ov_img, cax=cbar_ax, label='Elevation (m)')

        # Interactive selectors
        self.selector_ov = RectangleSelector(self.ax_rgb_ov, self._on_select, interactive=True)
        self.selector_zoom = RectangleSelector(self.ax_rgb_zoom, self._on_select, interactive=True)
        
        # Control Widgets
        self._setup_controls()
        
    def _setup_controls(self):
        """Create sliders and text boxes for parameters."""
        # Y positions for rows of controls
        y_rows = [0.14, 0.09, 0.04]
        labels = ['Scale', 'X Offset', 'Y Offset']
        keys = ['scale', 'x_off', 'y_off']
        ranges = [(0.1, 5.0), (-5000, 5000), (-5000, 5000)]
        inits = [1.0, 0, 0]
        
        self.widgets = {}
        
        for i, (label, key, rng, init) in enumerate(zip(labels, keys, ranges, inits)):
            # Slider
            ax_sl = self.fig.add_axes([0.15, y_rows[i], 0.35, 0.03])
            sl = Slider(ax_sl, f'{label} ', rng[0], rng[1], valinit=init)
            sl.on_changed(lambda val, k=key: self._update_param(k, val, 'slider'))
            
            # TextBox
            ax_txt = self.fig.add_axes([0.55, y_rows[i], 0.1, 0.03])
            txt = TextBox(ax_txt, '', initial=str(init))
            txt.on_submit(lambda val, k=key: self._update_param(k, val, 'text'))
            
            self.widgets[f'{key}_sl'] = sl
            self.widgets[f'{key}_txt'] = txt
            
        # Export button
        ax_export = self.fig.add_axes([0.75, 0.04, 0.15, 0.08])
        self.btn_export = Button(ax_export, 'EXPORT DATA', color='lightgreen', hovercolor='lime')
        self.btn_export.on_clicked(self._on_export)
        
        # Information readout
        self.info_text = self.fig.text(0.75, 0.14, 'No selection.\nDraw on any RGB panel.', fontsize=9, verticalalignment='top')

    def _update_param(self, key, val, source):
        """Synchronize slider and text box, then update display."""
        try:
            val = float(val)
        except ValueError:
            return
            
        self.params[key] = val
        
        # Sync other widget
        if source == 'slider':
            self.widgets[f'{key}_txt'].set_val(f'{val:.4f}' if key == 'scale' else f'{int(val)}')
        else:
            self.widgets[f'{key}_sl'].set_val(val)
            
        self._update_display()

    def _on_select(self, eclick, erelease):
        """Handle rectangle selection on either RGB panel."""
        # Determine which axis was clicked
        if eclick.inaxes == self.ax_rgb_ov:
            px1, py1 = eclick.xdata, eclick.ydata
            px2, py2 = erelease.xdata, erelease.ydata
        elif eclick.inaxes == self.ax_rgb_zoom:
            px1, py1 = eclick.xdata, eclick.ydata
            px2, py2 = erelease.xdata, erelease.ydata
        else:
            return

        # Preview coordinates (clamped)
        px1, px2 = np.clip([px1, px2], 0, self.rgb_array.shape[1])
        py1, py2 = np.clip([py1, py2], 0, self.rgb_array.shape[0])
        
        self.preview_selection = (min(px1, px2), min(py1, py2), max(px1, px2), max(py1, py2))
        
        # Convert to full-res coordinates
        fx1 = int(self.preview_selection[0] / self.rgb_scale)
        fy1 = int(self.preview_selection[1] / self.rgb_scale)
        fx2 = int(self.preview_selection[2] / self.rgb_scale)
        fy2 = int(self.preview_selection[3] / self.rgb_scale)
        self.selection = (fx1, fy1, fx2, fy2)
        
        self._update_display()

    def _update_display(self):
        """Update rectangles, zoom limits, and text."""
        if self.preview_selection is None:
            return
            
        px1, py1, px2, py2 = self.preview_selection
        pw = px2 - px1
        ph = py2 - py1
        
        # 1. Update RGB Overview rectangle
        self.rect_rgb_ov.set_xy((px1, py1))
        self.rect_rgb_ov.set_width(pw)
        self.rect_rgb_ov.set_height(ph)
        
        # 2. Update Elevation Overview rectangle (calculated based on parameters)
        # Convert preview RGB to full-res
        fx1, fy1 = px1/self.rgb_scale, py1/self.rgb_scale
        fx2, fy2 = px2/self.rgb_scale, py2/self.rgb_scale
        
        # Apply transformation to get full-res elevation coords
        s, xoff, yoff = self.params['scale'], self.params['x_off'], self.params['y_off']
        fex1, fey1 = fx1 * s + xoff, fy1 * s + yoff
        fex2, fey2 = fx2 * s + xoff, fy2 * s + yoff
        
        # Convert to elevation preview space
        pex1, pey1 = fex1 * self.elev_scale, fey1 * self.elev_scale
        pex2, pey2 = fex2 * self.elev_scale, fey2 * self.elev_scale
        
        self.rect_elev_ov.set_xy((pex1, pey1))
        self.rect_elev_ov.set_width(pex2 - pex1)
        self.rect_elev_ov.set_height(pey2 - pey1)
        
        # 3. Zoom into RGB panel
        # Add padding to zoom view
        pad_x = pw * 0.2
        pad_y = ph * 0.2
        self.ax_rgb_zoom.set_xlim(px1 - pad_x, px2 + pad_x)
        self.ax_rgb_zoom.set_ylim(py2 + pad_y, py1 - pad_y) # Inverted for image
        
        # 4. Zoom into Elevation panel
        epw = pex2 - pex1
        eph = pey2 - pey1
        epad_x = epw * 0.2
        epad_y = eph * 0.2
        self.ax_elev_zoom.set_xlim(pex1 - epad_x, pex2 + epad_x)
        self.ax_elev_zoom.set_ylim(pey2 + epad_y, pey1 - epad_y)
        
        # 5. Update info text
        self.info_text.set_text(
            f'RGB Size: {int(fx2-fx1)}x{int(fy2-fy1)}\n'
            f'ELV Size: {int(fex2-fex1)}x{int(fey2-fey1)}\n'
            f'Coord: ({int(fx1)}, {int(fy1)})'
        )
        
        self.fig.canvas.draw_idle()

    def _on_export(self, event):
        """Save full-resolution crops."""
        if self.selection is None:
            return
            
        print("\nExporting full-resolution crops...")
        fx1, fy1, fx2, fy2 = self.selection
        
        # Save RGB
        rgb_path = self.output_dir / "rgb_cropped.jpg"
        rgb_crop = self.rgb_image.crop((fx1, fy1, fx2, fy2))
        rgb_crop.save(rgb_path, quality=95)
        print(f"  Saved RGB to {rgb_path}")
        
        # Save Elevation
        s, xoff, yoff = self.params['scale'], self.params['x_off'], self.params['y_off']
        fex1, fey1 = int(fx1 * s + xoff), int(fy1 * s + yoff)
        fex2, fey2 = int(fx2 * s + xoff), int(fy2 * s + yoff)
        
        # Clamp to bounds
        fex1 = max(0, min(fex1, self.elevation_data.shape[1]-1))
        fey1 = max(0, min(fey1, self.elevation_data.shape[0]-1))
        fex2 = max(0, min(fex2, self.elevation_data.shape[1]))
        fey2 = max(0, min(fey2, self.elevation_data.shape[0]))
        
        elev_crop = self.elevation_data[fey1:fey2, fex1:fex2]
        elev_path = self.output_dir / "elevation_cropped.asc"
        self._write_asc(elev_path, elev_crop, fex1, fey1)
        print(f"  Saved Elevation to {elev_path}")
        print("✓ Done!")

    def _write_asc(self, path, data, x_off, y_off):
        """Write ESRI ASCII Grid file."""
        nrows, ncols = data.shape
        cellsize = self.elev_header['cellsize']
        xll = self.elev_header['xllcorner'] + (x_off * cellsize)
        yll = self.elev_header['yllcorner'] + ((self.elev_header['nrows'] - y_off - nrows) * cellsize)
        nodata = self.elev_header.get('nodata_value', -9999)
        
        with open(path, 'w') as f:
            f.write(f"ncols         {ncols}\n")
            f.write(f"nrows         {nrows}\n")
            f.write(f"xllcorner     {xll}\n")
            f.write(f"yllcorner     {yll}\n")
            f.write(f"cellsize      {cellsize}\n")
            f.write(f"NODATA_value  {nodata}\n")
            
            # Fill NaNs back to nodata
            out = np.copy(data)
            out[np.isnan(out)] = nodata
            np.savetxt(f, out, fmt='%.2f')

    def run(self):
        print("\nTool running. Interactions:")
        print(" 1. Draw rectangle on RGB OVERVIEW (top-left) or RGB ZOOM (bottom-left)")
        print(" 2. Adjust SCALE or OFFSETS to match the ELEVATION data features")
        print(" 3. Click EXPORT to save the aligned crops.")
        plt.show()

def main():
    base = Path(__file__).parent
    tool = OverlapExtractionTool(
        base / "RGBImagery" / "raw.jpg",
        base / "Elevation" / "sfbaydeltadem10m2016.asc"
    )
    tool.run()

if __name__ == "__main__":
    main()
