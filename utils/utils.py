import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import wandb
from torch.nn import CTCLoss

# Constantes globales
IMG_WIDTH = 128
IMG_HEIGHT = 32
CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:()'\"- "
CHAR2IDX = {ch: i+1 for i, ch in enumerate(CHARS)}  # 0 es blank
IDX2CHAR = {i+1: ch for i, ch in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1

def text_to_indices(text):
    return [CHAR2IDX[c] for c in text if c in CHAR2IDX]

def indices_to_text(indices):
    return ''.join([IDX2CHAR[idx] for idx in indices if idx in IDX2CHAR])

class IAMDataset(Dataset):
    def __init__(self, gt_file, img_dir):
        self.img_dir = img_dir
        self.samples = []
        with open(gt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                img_path, label = line.split('\t')
                full_path = os.path.join(img_dir, img_path)
                self.samples.append((full_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert('L')
        except:
            img = Image.new('L', (IMG_WIDTH, IMG_HEIGHT), 0)
        img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.BILINEAR)
        img = np.array(img, dtype=np.float32) / 255.0
        # Normalización: pasar a rango [-1, 1] (media ~0.5, std ~0.5)
        img = (img - 0.5) / 0.5
        img = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)
        label_indices = text_to_indices(label)
        label_len = len(label_indices)
        return img, torch.tensor(label_indices, dtype=torch.int), label_len

def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    labels = [item[1] for item in batch]
    label_lengths = [item[2] for item in batch]
    max_len = max(label_lengths)
    padded_labels = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, (lab, l) in enumerate(zip(labels, label_lengths)):
        padded_labels[i, :l] = lab
    return images, padded_labels, torch.tensor(label_lengths)

def make_loaders(train_gt, val_gt, test_gt, img_dir, batch_size):
    train_dataset = IAMDataset(train_gt, img_dir)
    val_dataset = IAMDataset(val_gt, img_dir)
    test_dataset = IAMDataset(test_gt, img_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=2, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=2, collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=2, collate_fn=collate_fn, pin_memory=True)
    return train_loader, val_loader, test_loader

def make(config, device):
    from models import CRNN  # importar aquí para evitar circular
    
    train_loader, val_loader, test_loader = make_loaders(
        config.train_gt, config.val_gt, config.test_gt, config.img_dir, config.batch_size
    )
    
    model = CRNN(NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH, hidden_size=config.hidden_size).to(device)
    criterion = CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    return model, train_loader, val_loader, test_loader, criterion, optimizer