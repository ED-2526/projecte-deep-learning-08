import torch
from utils.utils import IAMDataset, indices_to_text, NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH, collate_fn
from models import CRNN
from torch.utils.data import DataLoader
import Levenshtein

def test(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dataset = IAMDataset(config.test_gt, config.img_dir)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                             num_workers=2, collate_fn=collate_fn)
    
    model = CRNN(NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH, hidden_size=config.hidden_size).to(device)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
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
                cer = Levenshtein.distance(pred_text, real_text) / max(len(real_text), 1)
                total_cer += cer
                total_samples += 1
    avg_cer = total_cer / total_samples
    print(f"Character Error Rate en test: {avg_cer:.4f}")