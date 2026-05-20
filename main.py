import argparse
import train
import test
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
GROUP_ROOT = PROJECT_ROOT.parents[1]

DATASET_CONFIGS = {
    "iam": {
        "wandb_project": "iam_handwriting",
        "checkpoint_path": "best_model.pth",
        "cer_checkpoint_path": "best_cer_model.pth",
        "train_gt": PROJECT_ROOT / "splits" / "train_gt_80.txt",
        "val_gt": PROJECT_ROOT / "splits" / "val_gt_10.txt",
        "test_gt": PROJECT_ROOT / "splits" / "test_gt_10.txt",
        "img_dir": Path("/home/datasets/iam_dataset"),
        "zip_path": None,
        "text_normalization": "none",
        "max_width": 512,
    },
    "esposalles": {
        "wandb_project": "esposalles_handwriting",
        "checkpoint_path": "best_esposalles_model.pth",
        "cer_checkpoint_path": "best_esposalles_cer_model.pth",
        "train_gt": PROJECT_ROOT / "splits" / "esposalles" / "train_gt_official.txt",
        "val_gt": PROJECT_ROOT / "splits" / "esposalles" / "val_gt_official.txt",
        "test_gt": PROJECT_ROOT / "splits" / "esposalles" / "test_gt_official.txt",
        "img_dir": PROJECT_ROOT / "data" / "esposalles",
        "zip_path": None,
        "text_normalization": "esposalles",
        "max_width": 768,
    },
}


def apply_dataset_config(config, dataset_name):
    for key, value in DATASET_CONFIGS[dataset_name].items():
        setattr(config, key, str(value) if isinstance(value, Path) else value)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train')
    parser.add_argument('--dataset', type=str, choices=DATASET_CONFIGS.keys(), default='iam')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--max-width', type=int, default=None)
    parser.add_argument('--learning-rate', type=float, default=None)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--wandb-mode', type=str, choices=['online', 'offline', 'disabled'], default=None)
    parser.add_argument('--wandb-project', type=str, default=None)
    parser.add_argument('--wandb-name', type=str, default=None)
    parser.add_argument('--max-train-batches', type=int, default=None)
    parser.add_argument('--max-val-batches', type=int, default=None)
    parser.add_argument('--prediction-samples', type=int, default=None)
    parser.add_argument('--checkpoint-path', type=str, default=None)
    parser.add_argument('--cer-checkpoint-path', type=str, default=None)
    parser.add_argument('--train-gt', type=str, default=None)
    parser.add_argument('--val-gt', type=str, default=None)
    parser.add_argument('--test-gt', type=str, default=None)
    parser.add_argument('--img-dir', type=str, default=None)
    parser.add_argument('--zip-path', type=str, default=None)
    parser.add_argument('--text-normalization', type=str, choices=['none', 'esposalles'], default=None)
    parser.add_argument(
        '--image-preprocess',
        type=str,
        choices=['none', 'autocontrast', 'binarize', 'autocontrast_binarize'],
        default=None,
    )
    parser.add_argument('--test-samples', type=int, default=None)
    parser.add_argument('--test-output', type=str, default=None)
    parser.add_argument('--test-progress-every', type=int, default=None)
    parser.add_argument('--quiet-test-samples', action='store_true')
    args = parser.parse_args()
    
    # Configuración con rutas absolutas
    class Config:
        batch_size = 32
        num_workers = 2
        learning_rate = 0.0005          # tasa más conservadora
        epochs = 100                    # muchas épocas, early stopping lo cortará
        hidden_size = 256
        dropout = 0.2
        max_width = 512
        image_preprocess = "none"
        patience = 20                   # más paciencia
        wandb_project = "iam_handwriting"
        wandb_mode = "online"
        wandb_name = None
        max_train_batches = None
        max_val_batches = None
        prediction_samples = 8
        checkpoint_path = None
        cer_checkpoint_path = None
        test_samples = 0
        test_output = None
        test_progress_every = 0
        quiet_test_samples = False
        dataset = args.dataset
        train_gt = None
        val_gt = None
        test_gt = None
        img_dir = None
        zip_path = None
        text_normalization = "none"
    
    # Verificar que los archivos existen (opcional pero recomendado)
    config = Config()
    apply_dataset_config(config, args.dataset)
    overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "max_width": args.max_width,
        "learning_rate": args.learning_rate,
        "patience": args.patience,
        "wandb_mode": args.wandb_mode,
        "wandb_project": args.wandb_project,
        "wandb_name": args.wandb_name,
        "max_train_batches": args.max_train_batches,
        "max_val_batches": args.max_val_batches,
        "prediction_samples": args.prediction_samples,
        "checkpoint_path": args.checkpoint_path,
        "cer_checkpoint_path": args.cer_checkpoint_path,
        "train_gt": args.train_gt,
        "val_gt": args.val_gt,
        "test_gt": args.test_gt,
        "img_dir": args.img_dir,
        "zip_path": args.zip_path,
        "text_normalization": args.text_normalization,
        "image_preprocess": args.image_preprocess,
        "test_samples": args.test_samples,
        "test_output": args.test_output,
        "test_progress_every": args.test_progress_every,
        "quiet_test_samples": args.quiet_test_samples,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(config, key, value)

    for path in [config.train_gt, config.val_gt, config.test_gt, config.img_dir, config.zip_path]:
        if path is None:
            continue
        if not os.path.exists(path):
            print(f"Error: No se encuentra {path}")
            return
    
    if args.mode == 'train':
        train.train(config)
    else:
        test.test(config)

if __name__ == '__main__':
    main()
