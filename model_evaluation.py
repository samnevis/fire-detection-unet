import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import rasterio
from tqdm import tqdm
from model_training import UNet, WildfireDataset
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
from sklearn.model_selection import train_test_split

def load_model(model_path='best_model.pth', device='cuda'):
    """Load the trained model"""
    model = UNet()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    return model

def evaluate_model(model, test_loader, device='cuda'):
    """Evaluate model performance on test set"""
    predictions = []
    ground_truths = []
    
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            predictions.append(outputs.cpu().numpy())
            ground_truths.append(masks.cpu().numpy())
    
    predictions = np.concatenate(predictions)
    ground_truths = np.concatenate(ground_truths)
    
    # Convert to binary predictions
    binary_preds = (predictions > 0.5).astype(np.float32)
    
    # Calculate metrics
    precision = precision_score(ground_truths.flatten(), binary_preds.flatten())
    recall = recall_score(ground_truths.flatten(), binary_preds.flatten())
    f1 = f1_score(ground_truths.flatten(), binary_preds.flatten())
    iou = jaccard_score(ground_truths.flatten(), binary_preds.flatten())
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou
    }, predictions, ground_truths

def visualize_predictions(predictions, ground_truths, test_images, save_dir='evaluation_results', threshold=0.5):
    """Visualize and save model predictions"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    for i in range(min(5, len(predictions))):  # Visualize first 5 predictions
        pred = predictions[i].squeeze()
        gt = ground_truths[i].squeeze()
        
        # Threshold predictions to binary values
        binary_pred = (pred > threshold).astype(np.float32)
        
        # Load original image
        with rasterio.open(test_images[i]) as src:
            image = src.read()
            image = np.transpose(image[:3], (1, 2, 0))  # Get RGB channels
            image = (image - image.min()) / (image.max() - image.min())
        
        # Create visualization
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
        
        ax1.imshow(image)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        ax2.imshow(gt, cmap='gray')
        ax2.set_title('Ground Truth')
        ax2.axis('off')
        
        ax3.imshow(pred, cmap='gray')
        ax3.set_title('Raw Prediction (Probabilities)')
        ax3.axis('off')
        
        ax4.imshow(binary_pred, cmap='gray')
        ax4.set_title('Thresholded Prediction')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_dir / f'prediction_{i}.png')
        plt.close()

def main():
    # Set up paths
    data_path = Path("data")
    images_path = data_path / "images"
    masks_path = data_path / "masks"
    
    # Get all image and mask paths
    image_files = sorted(list(images_path.glob("*.tif")))
    mask_files = sorted(list(masks_path.glob("*.tif")))
    
    # Split data into train+val and test sets using the same random seed as training
    # First split: 80% train+val, 20% test
    train_val_images, test_images, train_val_masks, test_masks = train_test_split(
        image_files, mask_files, test_size=0.2, random_state=42
    )
    
    # Use only first 5 test images for visualization
    test_images = test_images[:5]
    test_masks = test_masks[:5]
    
    # Create dataset with test images
    test_dataset = WildfireDataset(test_images, test_masks)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    model = load_model(device=device)
    
    # Evaluate model
    metrics, predictions, ground_truths = evaluate_model(model, test_loader, device)
    
    # Print metrics
    print("\nModel Performance Metrics on Test Set:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"IoU Score: {metrics['iou']:.4f}")
    
    # Visualize predictions
    print("\nSaving prediction visualizations...")
    visualize_predictions(predictions, ground_truths, test_images)
    print("Visualizations saved to 'evaluation_results' directory")

if __name__ == "__main__":
    main() 