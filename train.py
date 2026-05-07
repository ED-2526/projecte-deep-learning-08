import torch
import wandb
from utils import make, indices_to_text

try:
    import Levenshtein
except ImportError:
    Levenshtein = None

def config_to_dict(config):
    return {
        key: getattr(config, key)
        for key in dir(config)
        if not key.startswith("_") and not callable(getattr(config, key))
    }

def train_one_epoch(model, loader, optimizer, criterion, device, max_batches=None):
    model.train()
    total_loss = 0
    total_batches = 0
    for batch_idx, (images, labels, label_lengths) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)
        outputs = model(images)  # (T, batch, num_classes)
        T = outputs.size(0)
        batch_size = outputs.size(1)
        input_lengths = torch.full((batch_size,), T, dtype=torch.long, device=device)
        loss = criterion(outputs.log_softmax(2), labels, input_lengths, label_lengths)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
        total_batches += 1
    return total_loss / max(total_batches, 1)

def decode_prediction(pred_indices, sample_idx):
    pred_seq = []
    prev = -1
    for t in range(pred_indices.size(0)):
        idx = pred_indices[t, sample_idx].item()
        if idx != prev and idx != 0:
            pred_seq.append(idx)
        prev = idx
    return pred_seq, indices_to_text(pred_seq)

def levenshtein_distance(a, b):
    if Levenshtein is not None:
        return Levenshtein.distance(a, b)

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (ca != cb),
            ))
        previous = current
    return previous[-1]

def calculate_cer(pred_text, real_text):
    return levenshtein_distance(pred_text, real_text) / max(len(real_text), 1)

def image_for_wandb(image_tensor):
    image = image_tensor.detach().cpu().squeeze(0)
    image = (image * 0.5 + 0.5).clamp(0, 1)
    return wandb.Image(image.numpy())

def make_prediction_table(examples):
    table = wandb.Table(columns=["epoch", "image", "target", "prediction", "cer"])
    for example in examples:
        table.add_data(
            example["epoch"],
            example["image"],
            example["target"],
            example["prediction"],
            example["cer"],
        )
    return table

def validate(model, loader, criterion, device, epoch=None, max_batches=None, prediction_samples=8):
    model.eval()
    total_loss = 0
    total_batches = 0
    total_edit_distance = 0
    total_target_chars = 0
    prediction_examples = []
    with torch.no_grad():
        for i, (images, labels, label_lengths) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break

            images_cpu = images.detach().cpu()
            images = images.to(device)
            labels = labels.to(device)
            label_lengths = label_lengths.to(device)
            outputs = model(images)
            T = outputs.size(0)
            batch_size = outputs.size(1)
            input_lengths = torch.full((batch_size,), T, dtype=torch.long, device=device)
            loss = criterion(outputs.log_softmax(2), labels, input_lengths, label_lengths)
            total_loss += loss.item()
            total_batches += 1

            pred_indices = torch.argmax(outputs, dim=2)  # (T, batch)

            decoded_batch = []
            real_batch = []
            for sample_idx in range(batch_size):
                _, pred_text = decode_prediction(pred_indices, sample_idx)
                real_seq = labels[sample_idx, :label_lengths[sample_idx].item()].tolist()
                real_text = indices_to_text(real_seq)
                decoded_batch.append(pred_text)
                real_batch.append(real_text)
                total_edit_distance += levenshtein_distance(pred_text, real_text)
                total_target_chars += len(real_text)

            if epoch is not None and i < 3 and batch_size > 0:
                pred_seq, _ = decode_prediction(pred_indices, 0)
                pred_text = decoded_batch[0]
                real_text = real_batch[0]
                print(f"  [Ejemplo] Predicción: {pred_text} | Real: {real_text}")

                if i == 0:
                    print(f"    [Debug] primeros 20 índices predichos: {pred_seq[:20]}")

            while len(prediction_examples) < prediction_samples:
                sample_idx = len(prediction_examples) % batch_size
                if sample_idx >= batch_size:
                    break

                pred_text = decoded_batch[sample_idx]
                real_text = real_batch[sample_idx]
                prediction_examples.append({
                    "epoch": epoch + 1 if epoch is not None else None,
                    "image": image_for_wandb(images_cpu[sample_idx]),
                    "target": real_text,
                    "prediction": pred_text,
                    "cer": calculate_cer(pred_text, real_text),
                })

                if len(prediction_examples) % batch_size == 0:
                    break

    val_loss = total_loss / max(total_batches, 1)
    val_cer = total_edit_distance / max(total_target_chars, 1)
    return val_loss, val_cer, prediction_examples

def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    model, train_loader, val_loader, _, criterion, optimizer = make(config, device)

    # Scheduler para reducir LR cuando la pérdida de validación se estanque
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5,
    )

    wandb_mode = getattr(config, "wandb_mode", "online")
    wandb_project = getattr(config, "wandb_project", "iam_handwriting")
    wandb_name = getattr(config, "wandb_name", None)
    max_train_batches = getattr(config, "max_train_batches", None)
    max_val_batches = getattr(config, "max_val_batches", None)
    prediction_samples = getattr(config, "prediction_samples", 8)
    checkpoint_path = getattr(config, "checkpoint_path", "best_model.pth")

    best_val_loss = float('inf')
    patience_counter = 0
    patience = getattr(config, 'patience', 20)  # paciencia alta

    with wandb.init(
        project=wandb_project,
        name=wandb_name,
        config=config_to_dict(config),
        mode=wandb_mode,
    ) as run:
        if wandb_mode != "disabled":
            wandb.watch(model, criterion, log="gradients", log_freq=100)

        for epoch in range(config.epochs):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device,
                max_batches=max_train_batches,
            )
            val_loss, val_cer, prediction_examples = validate(
                model, val_loader, criterion, device, epoch=epoch,
                max_batches=max_val_batches,
                prediction_samples=prediction_samples if wandb_mode != "disabled" else 0,
            )
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch+1}: train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, val_cer={val_cer:.4f}, lr={current_lr:.2e}"
            )

            log_data = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_cer": val_cer,
                "learning_rate": current_lr,
                "best_val_loss": min(best_val_loss, val_loss),
            }
            if wandb_mode != "disabled" and prediction_examples:
                log_data["validation_predictions"] = make_prediction_table(prediction_examples)
            wandb.log(log_data)

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  -> Mejor modelo guardado (val_loss={val_loss:.4f})")

                if wandb_mode != "disabled":
                    wandb.save(checkpoint_path)
                    artifact = wandb.Artifact("best_model", type="model")
                    artifact.add_file(checkpoint_path)
                    run.log_artifact(artifact)
            else:
                patience_counter += 1
                print(f"  -> Sin mejora ({patience_counter}/{patience})")
                if patience_counter >= patience:
                    print(f"Early stopping activado después de {epoch+1} épocas")
                    break

    print("Entrenamiento completado. Mejor pérdida de validación: {:.4f}".format(best_val_loss))

# Configuración por defecto (ajusta rutas según tu sistema)
if __name__ == "__main__":
    class Config:
        batch_size = 32
        learning_rate = 0.0005
        epochs = 100
        hidden_size = 256
        dropout = 0.2
        max_width = 512
        patience = 20
        wandb_project = "iam_handwriting"
        wandb_mode = "online"
        wandb_name = None
        max_train_batches = None
        max_val_batches = None
        prediction_samples = 8
        checkpoint_path = "best_model.pth"
        train_gt = "/home/edxnG08/projecte-deep-learning-08/grup_8/iam_dataset/train_gt.txt"
        val_gt = "/home/edxnG08/projecte-deep-learning-08/grup_8/iam_dataset/val_gt.txt"
        test_gt = "/home/edxnG08/projecte-deep-learning-08/grup_8/iam_dataset/linux_gt.txt"
        img_dir = "/home/edxnG08/projecte-deep-learning-08/grup_8/iam_dataset"
        zip_path = "/home/edxnG08/projecte-deep-learning-08/grup_8/iam_dataset.zip"
    config = Config()
    train(config)
