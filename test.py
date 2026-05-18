import csv

import torch
import wandb
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

def make_test_prediction_table(examples):
    table = wandb.Table(columns=[
        "index", "image", "target", "prediction", "cer", "word_correct", "word_error",
    ])
    for example in examples:
        table.add_data(
            example["index"],
            example["image"],
            example["target"],
            example["prediction"],
            example["cer"],
            example["word_correct"],
            example["word_error"],
        )
    return table

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
    correct_words = 0
    examples = []
    requested_examples = getattr(config, "test_samples", 0)
    output_path = getattr(config, "test_output", None)
    progress_every = getattr(config, "test_progress_every", 0)
    quiet_examples = getattr(config, "quiet_test_samples", False)
    wandb_mode = getattr(config, "wandb_mode", "disabled")
    wandb_project = getattr(config, "wandb_project", "iam_handwriting")
    wandb_name = getattr(config, "wandb_name", None)

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
                word_correct = pred_text == real_text
                word_error = not word_correct
                correct_words += int(word_correct)
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
                        "word_correct": word_correct,
                        "word_error": word_error,
                    })

            if progress_every and batch_idx % progress_every == 0:
                partial_cer = total_edit_distance / max(total_target_chars, 1)
                partial_word_accuracy = correct_words / max(total_samples, 1)
                partial_wer = 1.0 - partial_word_accuracy
                print(
                    f"Procesados {total_samples} ejemplos - "
                    f"CER parcial: {partial_cer:.4f} - "
                    f"word_accuracy parcial: {partial_word_accuracy:.4f} - "
                    f"WER parcial: {partial_wer:.4f}"
                )

    cer_global = total_edit_distance / max(total_target_chars, 1)
    word_accuracy = correct_words / max(total_samples, 1)
    wer = 1.0 - word_accuracy

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
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "index", "image", "target", "prediction",
                    "cer", "word_correct", "word_error",
                ],
            )
            writer.writeheader()
            writer.writerows(examples)
        print(f"\nPredicciones guardadas en: {output_path}")

    print(f"Character Error Rate en test: {cer_global:.4f}")
    print(f"Word Accuracy en test: {word_accuracy:.4f}")
    print(f"Word Error Rate en test: {wer:.4f}")

    if wandb_mode != "disabled":
        run_name = wandb_name or f"test_{checkpoint_path}"
        with wandb.init(
            project=wandb_project,
            name=run_name,
            config={
                "mode": "test",
                "checkpoint_path": checkpoint_path,
                "test_gt": config.test_gt,
                "img_dir": config.img_dir,
                "test_samples": requested_examples,
            },
            mode=wandb_mode,
        ):
            log_data = {
                "test_cer": cer_global,
                "test_word_accuracy": word_accuracy,
                "test_wer": wer,
                "test_total_samples": total_samples,
            }
            if examples:
                log_data["test_predictions"] = make_test_prediction_table(examples)
            wandb.log(log_data)
