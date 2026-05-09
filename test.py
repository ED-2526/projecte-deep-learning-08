import csv

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
    
    total_edit_distance = 0
    total_target_chars = 0
    total_samples = 0
    examples = []
    requested_examples = getattr(config, "test_samples", 0)
    output_path = getattr(config, "test_output", None)
    progress_every = getattr(config, "test_progress_every", 0)
    quiet_examples = getattr(config, "quiet_test_samples", False)

    with torch.no_grad():
        for batch_idx, (images, labels, label_lengths) in enumerate(test_loader, start=1):
            images = images.to(device)
            outputs = model(images)
            pred_indices = torch.argmax(outputs, dim=2)
            for i in range(images.size(0)):
                sample_index = total_samples
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
                edit_distance = levenshtein_distance(pred_text, real_text)
                cer = edit_distance / max(len(real_text), 1)
                total_edit_distance += edit_distance
                total_target_chars += len(real_text)
                total_samples += 1
                #print(f"Target='{real_text}', Prediction='{pred_text}'")
                if len(examples) < requested_examples:
                    _, image_ref, _ = test_dataset.samples[sample_index]
                    examples.append({
                        "index": sample_index,
                        "image": image_ref,
                        "target": real_text,
                        "prediction": pred_text,
                        "cer": cer,
                    })

            if progress_every and batch_idx % progress_every == 0:
                partial_cer = total_edit_distance / max(total_target_chars, 1)
                print(f"Procesados {total_samples} ejemplos - CER parcial: {partial_cer:.4f}")

    cer_global = total_edit_distance / max(total_target_chars, 1)

    if examples and not quiet_examples:
        print("\nEjemplos de prediccion en test:")
        for example in examples:
            status = "OK" if example["cer"] == 0 else "ERR"
            print(
                f"[{status}] #{example['index']} "
                f"target='{example['target']}' "
                f"prediction='{example['prediction']}' "
                f"cer={example['cer']:.4f}"
            )
            print(f"      imagen: {example['image']}")

    if output_path:
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["index", "image", "target", "prediction", "cer"])
            writer.writeheader()
            writer.writerows(examples)
        print(f"\nPredicciones guardadas en: {output_path}")

    print(f"Character Error Rate en test: {cer_global:.4f}")
