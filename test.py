import torch
from utils.utils import IAMDataset, indices_to_text, NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH, collate_fn
from models import CRNN
from torch.utils.data import DataLoader

try:
    import Levenshtein
except ImportError:
    Levenshtein = None

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

def test(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dataset = IAMDataset(
        config.test_gt,
        config.img_dir,
        train=False,
        zip_path=getattr(config, "zip_path", None),
        max_width=getattr(config, "max_width", IMG_WIDTH),
    )
    if len(test_dataset) == 0:
        raise ValueError("El dataset de test esta vacio. Revisa rutas, txt e iam_dataset.zip.")

    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                             num_workers=2, collate_fn=collate_fn)
    
    model = CRNN(
        NUM_CLASSES,
        IMG_HEIGHT,
        IMG_WIDTH,
        hidden_size=config.hidden_size,
        dropout=getattr(config, "dropout", 0.2),
    ).to(device)
    checkpoint_path = getattr(config, "checkpoint_path", "best_model.pth")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    total_cer = 0.0
    total_samples = 0
    with torch.no_grad():
        for images, labels, label_lengths in test_loader:
            images = images.to(device)
            outputs = model(images)
            pred_indices = torch.argmax(outputs, dim=2)
            for i in range(images.size(0)):
                pred_seq = []
                prev = -1
                for t in range(pred_indices.size(0)):
                    idx = pred_indices[t, i].item()
                    if idx != prev and idx != 0:
                        pred_seq.append(idx)
                    prev = idx
                pred_text = indices_to_text(pred_seq)
                real_seq = labels[i, :label_lengths[i].item()].tolist()
                real_text = indices_to_text(real_seq)
                cer = levenshtein_distance(pred_text, real_text) / max(len(real_text), 1)
                total_cer += cer
                total_samples += 1
    avg_cer = total_cer / total_samples
    print(f"Character Error Rate en test: {avg_cer:.4f}")
