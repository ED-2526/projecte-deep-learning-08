import argparse
import train
import test
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
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
    args = parser.parse_args()
    
    # Configuración con rutas absolutas
    class Config:
        batch_size = 32
        learning_rate = 0.0005          # tasa más conservadora
        epochs = 100                    # muchas épocas, early stopping lo cortará
        hidden_size = 256
        dropout = 0.2
        max_width = 512
        patience = 20                   # más paciencia
        wandb_project = "iam_handwriting"
        wandb_mode = "online"
        wandb_name = None
        max_train_batches = None
        max_val_batches = None
        prediction_samples = 8
        checkpoint_path = "best_model.pth"
        cer_checkpoint_path = "best_cer_model.pth"
        train_gt = "/home/edxnG08/projecte-deep-learning-08/grup_8/iam_dataset/train_gt.txt"
        val_gt = "/home/edxnG08/projecte-deep-learning-08/grup_8/iam_dataset/val_gt.txt"
        test_gt = "/home/edxnG08/projecte-deep-learning-08/grup_8/iam_dataset/linux_gt.txt"
        img_dir = "/home/edxnG08/projecte-deep-learning-08/grup_8/iam_dataset"
        zip_path = "/home/edxnG08/projecte-deep-learning-08/grup_8/iam_dataset.zip"
    
    # Verificar que los archivos existen (opcional pero recomendado)
    config = Config()
    overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
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
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(config, key, value)

    for path in [config.train_gt, config.val_gt, config.test_gt, config.img_dir, config.zip_path]:
        if not os.path.exists(path):
            print(f"Error: No se encuentra {path}")
            return
    
    if args.mode == 'train':
        train.train(config)
    else:
        test.test(config)

if __name__ == '__main__':
    main()
