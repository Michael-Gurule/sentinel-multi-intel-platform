"""
Simple Training Script for OPIR CNN Classifier
Works with folder-based dataset structure
"""

import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.cnn_classifier import OPIREventCNN
from src.training.train_classifier import ModelTrainer


class FolderDataset(Dataset):
    """Load data from folder structure: train/class_name/*.npy"""
    
    def __init__(self, data_dir, split='train'):
        self.data_dir = Path(data_dir) / split
        self.class_names = ['launch', 'explosion', 'fire', 'aircraft', 'background']
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        # Collect all file paths
        self.samples = []
        for class_name in self.class_names:
            class_dir = self.data_dir / class_name
            for file_path in sorted(class_dir.glob('*.npy')):
                self.samples.append((file_path, self.class_to_idx[class_name]))
        
        print(f"  Loaded {len(self.samples)} samples from {split} set")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        signal = np.load(file_path)
        
        # Handle 2D arrays
        if signal.ndim == 2:
            signal = signal[0]
        
        # Normalize
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
        
        # Convert to tensor [1, time_steps]
        signal_tensor = torch.FloatTensor(signal).unsqueeze(0)
        label_tensor = torch.LongTensor([label])
        
        return signal_tensor, label_tensor


def main():
    """Main training function"""
    
    # ========== Configuration ==========
    DATA_DIR = 'data/synthetic/opir'
    OUTPUT_DIR = 'outputs/training'
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # Auto-detect device (MPS for Apple Silicon, CUDA for NVIDIA, CPU otherwise)
    if torch.backends.mps.is_available():
        DEVICE = 'mps'
    elif torch.cuda.is_available():
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'
    
    print("\n" + "="*60)
    print("SENTINEL OPIR CNN Classifier Training")
    print("="*60 + "\n")
    
    # ========== Load Data ==========
    print("Loading datasets...")
    train_dataset = FolderDataset(DATA_DIR, split='train')
    val_dataset = FolderDataset(DATA_DIR, split='validation')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    # Get input length from first sample
    sample_signal, _ = train_dataset[0]
    input_length = sample_signal.shape[1]
    
    print(f"\nDataset Configuration:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Input length: {input_length}")
    print(f"  Classes: {train_dataset.class_names}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Device: {DEVICE}")
    
    # ========== Initialize Model ==========
    print(f"\nInitializing CNN model...")
    model = OPIREventCNN(
        input_length=input_length,
        num_classes=5,  # 5 classes
        dropout_rate=0.3
    )
    
    # ========== Initialize Trainer ==========
    trainer = ModelTrainer(
        model=model,
        device=DEVICE,
        learning_rate=LEARNING_RATE
    )
    
    # ========== Train ==========
    print(f"\nStarting training...\n")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=EPOCHS,
        early_stopping_patience=10,
        checkpoint_dir=OUTPUT_DIR
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Best validation accuracy: {max(history['val_acc']):.2f}%")
    print(f"Model saved to: {OUTPUT_DIR}")
    print("="*60 + "\n")
    
    return history


if __name__ == '__main__':
    main()