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
    if Levenshtein is not None and isinstance(a, str) and isinstance(b, str):
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
        "index", "image", "target", "prediction", "cer",
        "line_correct", "line_error", "word_edit_distance", "word_error_rate",
    ])
    for example in examples:
        table.add_data(
            example["index"],
            example["image"],
            example["target"],
            example["prediction"],
            example["cer"],
            example["line_correct"],
            example["line_error"],
            example["word_edit_distance"],
            example["word_error_rate"],
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
        text_normalization=getattr(config, "text_normalization", "none"),
        image_preprocess=getattr(config, "image_preprocess", "none"),
    )
    if len(test_dataset) == 0:
        raise ValueError("El dataset de test esta vacio. Revisa rutas, txt e imagenes.")

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=getattr(config, "num_workers", 2),
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    
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
    correct_lines = 0
    total_word_edit_distance = 0
    total_target_words = 0
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
                line_correct = pred_text == real_text
                line_error = not line_correct
                correct_lines += int(line_correct)
                pred_words = pred_text.split()
                real_words = real_text.split()
                word_edit_distance = levenshtein_distance(pred_words, real_words)
                word_error_rate = word_edit_distance / max(len(real_words), 1)
                total_word_edit_distance += word_edit_distance
                total_target_words += len(real_words)
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
                        "word_correct": line_correct,
                        "word_error": line_error,
                        "line_correct": line_correct,
                        "line_error": line_error,
                        "word_edit_distance": word_edit_distance,
                        "word_error_rate": word_error_rate,
                    })

            if progress_every and batch_idx % progress_every == 0:
                partial_cer = total_edit_distance / max(total_target_chars, 1)
                partial_line_accuracy = correct_lines / max(total_samples, 1)
                partial_line_error_rate = 1.0 - partial_line_accuracy
                partial_word_level_wer = total_word_edit_distance / max(total_target_words, 1)
                print(
                    f"Procesados {total_samples} ejemplos - "
                    f"CER parcial: {partial_cer:.4f} - "
                    f"exact_line_accuracy parcial: {partial_line_accuracy:.4f} - "
                    f"line_error_rate parcial: {partial_line_error_rate:.4f} - "
                    f"word_level_WER parcial: {partial_word_level_wer:.4f}"
                )

    cer_global = total_edit_distance / max(total_target_chars, 1)
    line_accuracy = correct_lines / max(total_samples, 1)
    line_error_rate = 1.0 - line_accuracy
    word_level_wer = total_word_edit_distance / max(total_target_words, 1)

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
                    "line_correct", "line_error", "word_edit_distance", "word_error_rate",
                ],
            )
            writer.writeheader()
            writer.writerows(examples)
        print(f"\nPredicciones guardadas en: {output_path}")

    print(f"Character Error Rate en test: {cer_global:.4f}")
    print(f"Exact Line Accuracy en test: {line_accuracy:.4f}")
    print(f"Exact Line Error Rate en test: {line_error_rate:.4f}")
    print(f"Word-level WER en test: {word_level_wer:.4f}")

    if wandb_mode != "disabled":
        run_name = wandb_name or f"test_{checkpoint_path}"
        with wandb.init(
            project=wandb_project,
            name=run_name,
            config={
                "mode": "test",
                "checkpoint_path": checkpoint_path,
                "dataset": getattr(config, "dataset", None),
                "test_gt": config.test_gt,
                "img_dir": config.img_dir,
                "text_normalization": getattr(config, "text_normalization", "none"),
                "image_preprocess": getattr(config, "image_preprocess", "none"),
                "max_width": getattr(config, "max_width", IMG_WIDTH),
                "test_samples": requested_examples,
            },
            mode=wandb_mode,
        ):
            log_data = {
                "test_cer": cer_global,
                "test_exact_line_accuracy": line_accuracy,
                "test_exact_line_error_rate": line_error_rate,
                "test_word_level_wer": word_level_wer,
                "test_total_samples": total_samples,
            }
            if examples:
                log_data["test_predictions"] = make_test_prediction_table(examples)
            wandb.log(log_data)
