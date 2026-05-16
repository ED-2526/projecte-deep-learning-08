import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

class IAMWordsDataset(Dataset):
    def __init__(self, txt_file, img_root_dir, transform=None):
        """
        txt_file: Ruta a vuestro archivo (ej. 'train_gx.txt')
        img_root_dir: La carpeta base donde está la carpeta 'words/'
        """
        self.img_root_dir = img_root_dir
        self.transform = transform
        self.data = []
        self.vocab = set()

        # 1. Leer el archivo y extraer rutas y textos
        with open(txt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Separar por el tabulador
                parts = line.split('\t')
                if len(parts) == 2:
                    img_path = parts[0]
                    text = parts[1].lower()
                    
                    # --- NUEVO: Comprobamos si la imagen existe de verdad ---
                    # (Asumo que la carpeta principal la guardaste en self.img_dir)
                    img_full_path = os.path.join(self.img_root_dir, img_path)
                    
                    if os.path.exists(img_full_path):
                        self.data.append((img_path, text))
                        
                        # Añadir los caracteres al vocabulario SOLO si la imagen existe
                        self.vocab.update(list(text))
                    # --------------------------------------------------------

        # 2. Crear los diccionarios de conversión (Caracter <-> Número)
        # Importante: Reservamos el índice 0 para el token <BLANK> que necesita la CTC Loss
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(sorted(list(self.vocab)))}
        self.char_to_idx['<BLANK>'] = 0 
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 3. Obtener la ruta y el texto de la posición 'idx'
        img_rel_path, text = self.data[idx]
        # Asegúrate de usar el mismo nombre de variable que definiste en el __init__
        # Si usaste self.img_dir, cámbialo aquí:
        img_full_path = os.path.join(self.img_root_dir, img_rel_path)

        # 4. Cargar la imagen y pasarla a escala de grises
        image = Image.open(img_full_path).convert('L')

        # --- CAMBIO AQUÍ: Convertir siempre a Tensor ---
        if self.transform:
            image = self.transform(image)
        else:
            # Si no hay transformaciones, al menos la pasamos a Tensor
            from torchvision import transforms
            image = transforms.ToTensor()(image)
        # -----------------------------------------------

        # 5. Convertir el texto (string) a un Tensor de números
        label = [self.char_to_idx[c] for c in text]
        label_tensor = torch.tensor(label, dtype=torch.long)

        return image, label_tensor

def collate_fn(batch):
    """
    Recibe una lista de tuplas (imagen, etiqueta) que devuelve el Dataset.__getitem__
    """
    images, labels = zip(*batch)

    # 1. Padding de las Etiquetas (Textos)
    # pad_sequence rellena con nuestro <BLANK> (que le asignamos el 0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)
    
    # Guardamos las longitudes originales porque la función de pérdida (CTC Loss) lo necesita luego
    label_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)

    # 2. Padding de las Imágenes
    # Buscamos la altura y el ancho máximos en este lote específico
    max_height = max([img.shape[1] for img in images])
    max_width = max([img.shape[2] for img in images])
    
    padded_images = []
    for img in images:
        # Calculamos cuánto falta para llegar al máximo de cada dimensión
        pad_right = max_width - img.shape[2]
        pad_bottom = max_height - img.shape[1]
        
        # F.pad recibe (pad_izq, pad_der, pad_arriba, pad_abajo)
        # Rellenamos a la derecha y abajo para que todas midan lo mismo
        padded_img = F.pad(img, (0, pad_right, 0, pad_bottom), value=0) 
        padded_images.append(padded_img)
        
    # Apilamos todas las imágenes ya rellenas en un solo tensor
    images_padded = torch.stack(padded_images)
    
    return images_padded, labels_padded, label_lengths
