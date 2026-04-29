import torch
import wandb
from utils import make
import os
os.environ["WANDB_MODE"] = "disabled"  # Deshabilita wandb interactivo

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for images, labels, label_lengths in loader:
        images = images.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)
        outputs = model(images)
        T = outputs.size(0)
        batch_size = outputs.size(1)
        input_lengths = torch.full((batch_size,), T, dtype=torch.long, device=device)
        loss = criterion(outputs.log_softmax(2), labels, input_lengths, label_lengths)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, labels, label_lengths in loader:
            images = images.to(device)
            labels = labels.to(device)
            label_lengths = label_lengths.to(device)
            outputs = model(images)
            T = outputs.size(0)
            batch_size = outputs.size(1)
            input_lengths = torch.full((batch_size,), T, dtype=torch.long, device=device)
            loss = criterion(outputs.log_softmax(2), labels, input_lengths, label_lengths)
            total_loss += loss.item()
    return total_loss / len(loader)

def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, train_loader, val_loader, _, criterion, optimizer = make(config, device)
    
    wandb.init(project="iam_handwriting", config=config, mode="disabled")
    wandb.watch(model)
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = getattr(config, 'patience', 5)  # Número de epochs sin mejora antes de parar
    
    for epoch in range(config.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        wandb.log({"train_loss": train_loss, "val_loss": val_loss})
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            wandb.save("best_model.pth")
            print(f"  -> Mejor modelo guardado (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  -> Sin mejora ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"Early stopping activado después de {epoch+1} epochs")
                break
    
    print("Entrenamiento completado.")

if __name__ == "__main__":
    class Config:
        batch_size = 32
        learning_rate = 0.001
        epochs = 20
        hidden_size = 256
        patience = 5   #<-- Parámetro de early stopping. Si no mejora en 5 epochs se para.
        train_gt = "./iam_dataset/train_gt.txt"
        val_gt = "./iam_dataset/val_gt.txt"
        test_gt = "./iam_dataset/linux_gt.txt"
        img_dir = "./iam_dataset"
    config = Config()
    train(config)
