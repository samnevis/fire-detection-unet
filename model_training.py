import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import rasterio
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

class WildfireDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        
        # Print shape of first image to debug
        with rasterio.open(self.image_paths[0]) as src:
            first_image = src.read()
            print(f"First image shape: {first_image.shape}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        with rasterio.open(self.image_paths[idx]) as src:
            image = src.read()
        
        # Load mask
        with rasterio.open(self.mask_paths[idx]) as src:
            mask = src.read()
        
        # Convert to torch tensors
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()
        
        # Normalize image
        image = (image - image.min()) / (image.max() - image.min())
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder (reduced channels)
        self.enc1 = self.conv_block(in_channels, 32)  # Reduced from 64
        self.enc2 = self.conv_block(32, 64)          # Reduced from 128
        self.enc3 = self.conv_block(64, 128)         # Reduced from 256
        self.enc4 = self.conv_block(128, 256)        # Reduced from 512
        
        # Decoder (reduced channels)
        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(256, 128)  # 256 because we concatenate with enc3
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(128, 64)   # 128 because we concatenate with enc2
        
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(64, 32)    # 64 because we concatenate with enc1
        
        self.dec1 = nn.Conv2d(32, out_channels, kernel_size=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Decoder with skip connections
        dec4 = self.up4(enc4)
        dec4 = torch.cat([dec4, enc3], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.up3(dec4)
        dec3 = torch.cat([dec3, enc2], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.up2(dec3)
        dec2 = torch.cat([dec2, enc1], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.dec1(dec2)
        
        return torch.sigmoid(dec1)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_iterations=150, val_frequency=15, device='cuda'):
    model = model.to(device)
    best_val_loss = float('inf')
    iteration = 0
    
    # Create progress bar for total iterations
    pbar = tqdm(total=num_iterations, desc="Training")
    
    while iteration < num_iterations:
        model.train()
        for images, masks in train_loader:
            if iteration >= num_iterations:
                break
                
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            iteration += 1
            pbar.update(1)
            
            # Validation
            if iteration % val_frequency == 0:
                model.eval()
                val_loss = 0.0
                val_batches = 0
                with torch.no_grad():
                    # Only validate on a subset of the validation data
                    for val_images, val_masks in val_loader:
                        if val_batches >= 5:  # Only use 5 batches for validation
                            break
                        val_images, val_masks = val_images.to(device), val_masks.to(device)
                        val_outputs = model(val_images)
                        val_loss += criterion(val_outputs, val_masks).item()
                        val_batches += 1
                
                avg_val_loss = val_loss / val_batches
                print(f'\nIteration {iteration}/{num_iterations}')
                print(f'Validation Loss: {avg_val_loss:.4f}')
                
                # Save only if validation loss improves
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(model.state_dict(), 'best_model.pth')
                    print('Model saved! (New best validation loss)')
                
                model.train()
    
    pbar.close()

def main():
    # Set up paths
    data_path = Path("data")
    images_path = data_path / "images"
    masks_path = data_path / "masks"
    
    # Get all image and mask paths
    image_files = sorted(list(images_path.glob("*.tif")))
    mask_files = sorted(list(masks_path.glob("*.tif")))
    
    # Split into train and validation sets
    train_images, val_images, train_masks, val_masks = train_test_split(
        image_files, mask_files, test_size=0.2, random_state=42
    )
    
    # Create datasets
    train_dataset = WildfireDataset(train_images, train_masks)
    val_dataset = WildfireDataset(val_images, val_masks)
    
    # Create data loaders with larger batch size
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Initialize model, loss function, and optimizer
    model = UNet()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_iterations=150,    # Reduced from 500 to 150 (about 12 epochs)
                val_frequency=15,      # Validate every 15 iterations
                device=device)

if __name__ == "__main__":
    main() 