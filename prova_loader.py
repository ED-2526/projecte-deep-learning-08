import torch
from torch.utils.data import DataLoader

# Importamos la clase y la función que acabas de guardar
# Si lo guardaste en utils/dataset.py, sería: from utils.dataset import IAMWordsDataset, collate_fn
# Si lo tienes en el mismo directorio (ej. tu carpeta), usa:
from prova_dataset import IAMWordsDataset, collate_fn

def main():
    # 1. Definir las rutas (¡Ajusta estas rutas a donde estén los datos reales en el servidor!)
    txt_train = "/home/edxnG08/projecte-deep-learning-08/grup_8/iam_dataset/linux_gt.txt"  
    img_dir = "/home/edxnG08/projecte-deep-learning-08/grup_8/iam_dataset"       

    print("Cargando dataset...")
    # 2. Instanciar tu Dataset
    train_dataset = IAMWordsDataset(txt_file=txt_train, img_root_dir=img_dir)
    print(f"Total de imágenes en entrenamiento: {len(train_dataset)}")
    print(f"Tamaño del vocabulario: {len(train_dataset.vocab)}")

    # 3. Crear el DataLoader (Aquí conectamos PyTorch con tu código)
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,       # Pasaremos 32 imágenes a la red al mismo tiempo
        shuffle=True,        # Mezclamos los datos en cada época
        collate_fn=collate_fn # ¡Nuestra función mágica de padding!
    )

    # 4. Prueba de fuego: Extraer el primer Batch
    print("\nExtrayendo el primer lote (batch)...")
    for batch_idx, (images, labels, label_lengths) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        # Deberías ver algo como [32, 1, Alto, Ancho_Máximo_del_lote]
        print(f" - Forma del tensor de imágenes: {images.shape}") 
        # Deberías ver algo como [32, Longitud_Máxima_de_texto_del_lote]
        print(f" - Forma del tensor de etiquetas: {labels.shape}")
        # Rompemos el bucle porque solo queremos probar que el primero funciona
        break 

if __name__ == "__main__":
    main()