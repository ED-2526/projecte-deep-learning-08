import io
import os
import zipfile
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.nn.functional as F
from torch.nn import CTCLoss
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms

# Constantes globales
IMG_WIDTH = 512
IMG_HEIGHT = 32
CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:()'\"- "
CHAR2IDX = {ch: i+1 for i, ch in enumerate(CHARS)}  # 0 es blank
IDX2CHAR = {i+1: ch for i, ch in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1 #Blanket incluido

def text_to_indices(text):
    return [CHAR2IDX[c] for c in text if c in CHAR2IDX]

def indices_to_text(indices):
    return ''.join([IDX2CHAR[idx] for idx in indices if idx in IDX2CHAR])

class IAMDataset(Dataset):
    def __init__(self, gt_file, img_dir, train=True, zip_path=None, max_width=IMG_WIDTH):
        self.img_dir = Path(img_dir)
        self.zip_path = Path(zip_path) if zip_path else None
        self.max_width = max_width
        self.samples = []
        self._zip_file = None
        zip_names = set()

        if self.zip_path and self.zip_path.exists():
            with zipfile.ZipFile(self.zip_path) as zf:
                zip_names = set(zf.namelist())

        missing_count = 0
        with open(gt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) == 2:
                    img_path, label = parts
                else:
                    continue

                full_path = self.img_dir / img_path
                zip_member = f"iam_dataset/{img_path}"

                if full_path.exists():
                    self.samples.append(("file", str(full_path), label))
                elif zip_member in zip_names:
                    self.samples.append(("zip", zip_member, label))
                else:
                    missing_count += 1

        if missing_count > 0:
            print(f"Advertencia: {missing_count} muestras ignoradas porque la imagen no existe en {gt_file}")

        if train:
            self.transform = transforms.Compose([
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.samples)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_zip_file"] = None
        return state

    def _get_zip_file(self):
        if self._zip_file is None:
            self._zip_file = zipfile.ZipFile(self.zip_path)
        return self._zip_file

    def _load_image(self, source, img_ref):
        if source == "file":
            return Image.open(img_ref).convert('L')#Grayscale

        with self._get_zip_file().open(img_ref) as f:
            return Image.open(io.BytesIO(f.read())).convert('L')

    def __getitem__(self, idx):
        source, img_ref, label = self.samples[idx]
        try:
            img = self._load_image(source, img_ref)
        except Exception as e:
            print(f"Error cargando {img_ref}: {e}, usando imagen negra")
            img = Image.new('L', (IMG_HEIGHT * 4, IMG_HEIGHT), 0)

        original_width, original_height = img.size
        aspect_ratio = original_width / max(original_height, 1)
        label_indices = text_to_indices(label)
        min_width = max(1, len(label_indices) * 10)
        new_width = int(IMG_HEIGHT * aspect_ratio)
        new_width = min(max(new_width, min_width), self.max_width)
        img = img.resize((new_width, IMG_HEIGHT), Image.Resampling.BILINEAR)

        if self.transform:
            img = self.transform(img)

        img = np.array(img, dtype=np.float32) / 255.0
        img = (img - 0.5) / 0.5   # rango [-1, 1]
        img = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)

        return img, torch.tensor(label_indices, dtype=torch.long), len(label_indices)

def collate_fn(batch):
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    label_lengths = [item[2] for item in batch]

    max_width = max(img.shape[2] for img in images)
    padded_images = [
        F.pad(img, (0, max_width - img.shape[2], 0, 0), value=0)
        for img in images
    ]

    return (
        torch.stack(padded_images),
        pad_sequence(labels, batch_first=True, padding_value=0),
        torch.tensor(label_lengths, dtype=torch.long),
    )

def make_loaders(train_gt, val_gt, test_gt, img_dir, batch_size, zip_path=None, max_width=IMG_WIDTH):
    train_dataset = IAMDataset(train_gt, img_dir, train=True, zip_path=zip_path, max_width=max_width)
    val_dataset = IAMDataset(val_gt, img_dir, train=False, zip_path=zip_path, max_width=max_width)
    test_dataset = IAMDataset(test_gt, img_dir, train=False, zip_path=zip_path, max_width=max_width)

    for name, dataset in [("train", train_dataset), ("val", val_dataset), ("test", test_dataset)]:
        if len(dataset) == 0:
            raise ValueError(f"El dataset {name} esta vacio. Revisa rutas, txt e imagenes.")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=2, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=2, collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=2, collate_fn=collate_fn, pin_memory=True)
    return train_loader, val_loader, test_loader

def make(config, device):
    from models import CRNN  # import local para evitar circular

    zip_path = getattr(config, "zip_path", None)
    max_width = getattr(config, "max_width", IMG_WIDTH)
    dropout = getattr(config, "dropout", 0.2)

    train_loader, val_loader, test_loader = make_loaders(
        config.train_gt, config.val_gt, config.test_gt, config.img_dir,
        config.batch_size, zip_path=zip_path, max_width=max_width
    )
    
    model = CRNN(NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH,
                 hidden_size=config.hidden_size, dropout=dropout).to(device)
    criterion = CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    return model, train_loader, val_loader, test_loader, criterion, optimizer
