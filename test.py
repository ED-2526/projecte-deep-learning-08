import torch
import Levenshtein
from utils import make, IDX2CHAR

def decode_prediction(output):
    """Decodificación greedy: eliminar duplicados consecutivos y blanks (0)"""
    pred_indices = torch.argmax(output, dim=2)  # (T, batch) -> tomar batch 0
    pred_seq = []
    prev = -1
    for idx in pred_indices[:, 0]:  # solo primera muestra del batch? Mejor procesar todas
        idx = idx.item()
        if idx != prev and idx != 0:
            pred_seq.append(idx)
        prev = idx
    return ''.join([IDX2CHAR[i] for i in pred_seq])

def evaluate(model, loader, device):
    model.eval()
    total_cer = 0.0
    num_samples = 0
    with torch.no_grad():
        for images, labels, label_lengths in loader:
            images = images.to(device)
            outputs = model(images)  # (T, batch, classes)
            # Decodificar cada muestra del batch
            for i in range(images.size(0)):
                pred_seq = []
                prev = -1
                for t in range(outputs.size(0)):
                    idx = torch.argmax(outputs[t, i]).item()
                    if idx != prev and idx != 0:
                        pred_seq.append(idx)
                    prev = idx
                pred_text = ''.join([IDX2CHAR[i] for i in pred_seq])
                real_seq = [labels[i, j] for j in range(label_lengths[i].item())]
                real_text = ''.join([IDX2CHAR[int(idx)] for idx in real_seq])
                cer = Levenshtein.distance(pred_text, real_text) / max(len(real_text), 1)
                total_cer += cer
                num_samples += 1
    return total_cer / num_samples

def test(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _, _, test_loader, _, _ = make(config, device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    cer = evaluate(model, test_loader, device)
    print(f"Character Error Rate en test: {cer:.4f}")
    return cer

if __name__ == "__main__":
    class Config:
        batch_size = 32
        learning_rate = 0.001
        hidden_size = 256
        train_gt = "./iam_dataset/train_gt.txt"
        val_gt = "./iam_dataset/val_gt.txt"
        test_gt = "./iam_dataset/linux_gt.txt"
        img_dir = "./iam_dataset"
    config = Config()
    test(config)