# XN-Project title (replace it by the title of your project)
Write here a short summary about your project. The text must include a short introduction and the targeted goals

## Code structure
You must create as many folders as you consider. You can use the proposed structure or replace it by the one in the base code that you use as starting point. Do not forget to add Markdown files as needed to explain well the code and how to use it.

## Example Code
The given code is a simple CNN example training on the MNIST dataset. It shows how to set up the [Weights & Biases](https://wandb.ai/site)  package to monitor how your network is learning, or not.

Before running the code you have to create a local environment with conda and activate it. The provided [environment.yml](https://github.com/DCC-UAB/XNAP-Project/environment.yml) file has all the required dependencies. Run the following command: ``conda env create --file environment.yml `` to create a conda environment with all the required dependencies and then activate it:
```
conda activate xnap-example
```

To run the example code:
```
python main.py
```

## IAM y Esposalles

El flujo de entrenamiento y test es el mismo para los dos datasets. Se elige con
`--dataset`.

Entrenar con IAM:
```
python main.py --mode train --dataset iam --wandb-mode online
```

Entrenar con Esposalles:
```
python main.py --mode train --dataset esposalles --wandb-mode online
```

Si se ejecuta en un entorno donde el `DataLoader` da problemas con procesos
hijos, se puede desactivar el multiprocessing:
```
python main.py --mode train --dataset esposalles --num-workers 0 --wandb-mode online
```

Probar el mejor modelo de Esposalles:
```
python main.py --mode test --dataset esposalles --checkpoint-path best_esposalles_model.pth --test-samples 12
```

Por defecto, Esposalles usa:
```
splits/esposalles/train_gt_80.txt
splits/esposalles/val_gt_10.txt
splits/esposalles/test_gt_10.txt
data/esposalles
```



## Contributors
Write here the name and UAB mail of the group members

Xarxes Neuronals i Aprenentatge Profund
Grau dÉnginyeria de Dades, 
UAB, 2026
