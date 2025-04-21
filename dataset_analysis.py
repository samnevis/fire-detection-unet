import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def setup_paths():
    """Set up the paths for the dataset"""
    dataset_path = Path("data")
    images_path = dataset_path / "images"
    masks_path = dataset_path / "masks"
    return images_path, masks_path

def load_image_and_mask(image_path, mask_path):
    """Load a single image and its corresponding mask"""
    with rasterio.open(image_path) as src:
        image = src.read()
        
    with rasterio.open(mask_path) as src:
        mask = src.read()
    
    return image, mask

def display_sample(image, mask, save_path=None):
    """Display a sample image and its mask"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display RGB image (assuming bands are in order)
    # Normalize the image data to 0-1 range for proper display
    rgb_image = np.transpose(image[:3], (1, 2, 0))
    rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
    ax1.imshow(rgb_image)
    ax1.set_title('Satellite Image')
    ax1.axis('off')
    
    # Display mask
    ax2.imshow(mask[0], cmap='gray')
    ax2.set_title('Burned Area Mask')
    ax2.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def analyze_dataset(images_path, masks_path, num_samples=10):
    """Analyze the dataset and generate statistics"""
    image_files = sorted(list(images_path.glob("*.tif")))
    mask_files = sorted(list(masks_path.glob("*.tif")))
    
    print(f"Number of images: {len(image_files)}")
    print(f"Number of masks: {len(mask_files)}")
    
    # Initialize statistics
    stats = {
        'image_shapes': [],
        'mask_shapes': [],
        'burned_areas': [],
        'total_pixels': [],
        'burned_percentages': []
    }
    
    # Analyze first few samples
    for i in tqdm(range(min(num_samples, len(image_files))), desc="Analyzing samples"):
        image, mask = load_image_and_mask(image_files[i], mask_files[i])
        
        # Store shapes
        stats['image_shapes'].append(image.shape)
        stats['mask_shapes'].append(mask.shape)
        
        # Calculate burned area statistics
        total_pixels = mask.shape[1] * mask.shape[2]
        burned_pixels = np.sum(mask == 1)
        burned_percentage = (burned_pixels / total_pixels) * 100
        
        stats['total_pixels'].append(total_pixels)
        stats['burned_areas'].append(burned_pixels)
        stats['burned_percentages'].append(burned_percentage)
        
        # Display sample
        print(f"\nSample {i+1}:")
        print(f"Image shape: {image.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Burned area percentage: {burned_percentage:.2f}%")
        
        # Save visualization
        save_path = f"sample_{i+1}_visualization.png"
        display_sample(image, mask, save_path)
    
    # Create summary DataFrame
    df_stats = pd.DataFrame({
        'Image Shape': stats['image_shapes'],
        'Mask Shape': stats['mask_shapes'],
        'Total Pixels': stats['total_pixels'],
        'Burned Pixels': stats['burned_areas'],
        'Burned Percentage': stats['burned_percentages']
    })
    
    print("\nDataset Summary Statistics:")
    print(df_stats.describe())
    
    return df_stats

def main():
    # Set up paths
    images_path, masks_path = setup_paths()
    
    # Analyze dataset
    stats = analyze_dataset(images_path, masks_path)
    
    # Save statistics to CSV
    stats.to_csv('dataset_statistics.csv', index=False)
    print("\nStatistics saved to 'dataset_statistics.csv'")

if __name__ == "__main__":
    main() 