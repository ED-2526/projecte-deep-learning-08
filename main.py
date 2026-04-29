import argparse
import train
import test
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train')
    args = parser.parse_args()
    
    # Configuración con rutas absolutas
    class Config:
        batch_size = 16
        learning_rate = 0.001
        epochs = 25
        hidden_size = 256
        train_gt = "/home/edxnG08/projecte-deep-learning-08/grup_8/iam_dataset/train_gt.txt"
        val_gt = "/home/edxnG08/projecte-deep-learning-08/grup_8/iam_dataset/val_gt.txt"
        test_gt = "/home/edxnG08/projecte-deep-learning-08/grup_8/iam_dataset/linux_gt.txt"
        img_dir = "/home/edxnG08/projecte-deep-learning-08/grup_8/iam_dataset"
    
    # Verificar que los archivos existen (opcional pero recomendado)
    config = Config()
    for path in [config.train_gt, config.val_gt, config.test_gt, config.img_dir]:
        if not os.path.exists(path):
            print(f"Error: No se encuentra {path}")
            return
    
    if args.mode == 'train':
        train.train(config)
    else:
        test.test(config)

if __name__ == '__main__':
    main()