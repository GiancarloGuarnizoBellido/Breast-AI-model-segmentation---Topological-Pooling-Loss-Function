import argparse
import yaml
from addict import Dict
import wandb
import os
from sklearn.model_selection import StratifiedKFold
import torch
import pandas as pd

from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import matplotlib.pyplot as plt
import cv2
import lightning as L
from model_lightning_seg import MyModel
from data_datamodule_seg import WSIDataModule
import random
import numpy as np

def train(dataset_train, dataset_test,conf,k_fold_value):
    #os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('medium')
    data_dir = conf.dataset.data_dir
    train_file = dataset_train
    dev_file = dataset_test
    test_file = dataset_test
    cache_data = conf.dataset.cache_data
    rescale_factor = conf.dataset.rescale_factor

    num = dev_file.split('.')[0].split('_')[-1]
    tb_exp_name = f'kfold_{k_fold_value}_{conf.dataset.experiment}_{num}'

    # Setting a random seed for reproducibility
    if conf.train_par.random_seed == 'default':
        random_seed = 2024
    else:
        random_seed = conf.train_par.random_seed
        
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    #print(dev_file)
    #print(data_dir)
    # Create a DataModule
    data_module = WSIDataModule(batch_size=conf.train_par.batch_size, workers=conf.train_par.workers, train_file=train_file, 
                                dev_file=dev_file,test_file=None, data_dir=data_dir, cache_data=cache_data)
    # Tamaño del batch de imágenes: torch.Size([1, 1, 128, 128, 128])
    # Tamaño del batch de etiquetas: torch.Size([1])

    data_module.prepare_data()
    data_module.setup(stage="fit")
    results_path = os.path.join(conf.train_par.results_path, conf.dataset.experiment)
    os.makedirs(results_path, exist_ok=True)
    conf.train_par.results_model_filename = os.path.join(results_path, f'{tb_exp_name}')
    #wandb logger
    wandb_logger = WandbLogger(project="3d_a_unet_seg_2", entity="giancarlo-guarnizo",config=conf, name=tb_exp_name)
    #wandb.run.name = f'{conf.dataset.experiment}_{name}'
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=conf.train_par.patience, verbose=True, mode="min")
    model_checkpoint = ModelCheckpoint(
        filename=conf.train_par.results_model_filename, monitor="val_loss", mode="min"
    )
    lightning_model =MyModel(model_opts=conf.model_opts, train_par=conf.train_par)

    trainer = L.Trainer(
        #max_epochs=conf.train_par.epochs, accelerator="auto", devices="auto", precision='bf16-mixed',logger=wandb_logger,callbacks=[early_stop_callback,model_checkpoint],
        max_epochs=conf.train_par.epochs, accelerator="auto", devices="auto",logger=wandb_logger,callbacks=[early_stop_callback,model_checkpoint],        
        default_root_dir=results_path#,log_every_n_steps=46
    )
    trainer.fit(model=lightning_model, datamodule=data_module)
    return trainer


def create_files(k_fold_value, conf):
    df_data = pd.read_csv("/media/stefano/Seagate Expansion Drive/rochester_luis_emilio/LIMICM2_emilio/giancarlo_model/prueba/nombre_paciente_random.csv")
    
    # Ruta base para guardar los archivos de cada fold
    base_path = "/media/stefano/Seagate Expansion Drive/rochester_luis_emilio/LIMICM2_emilio/giancarlo_model/Classification/data_csv_seg/data_kfold_10"
    os.makedirs(base_path, exist_ok=True)
    
    # Combinar las columnas label y malign para crear una etiqueta compuesta
    df_data["combined_label"] = df_data["label"].astype(str) + '_' + df_data["malign"].astype(str)
    
    # Dividir los pacientes en k grupos para el k-fold
    skf = StratifiedKFold(n_splits=k_fold_value, shuffle=False)
    i = 0
    
    # Iterar sobre cada fold
    for fold, (train_index, test_index) in enumerate(skf.split(df_data["id"], df_data["combined_label"])):        
        # Obtener los pacientes de entrenamiento y prueba para este fold
        train_data = df_data.iloc[train_index].copy()  # Crear una copia explícita
        test_data = df_data.iloc[test_index].copy()    # Crear una copia explícita
        
        # Eliminar la columna combined_label sin usar inplace
        train_data = train_data.drop(columns=["combined_label"])
        test_data = test_data.drop(columns=["combined_label"])
        
        # Crear los nombres de los archivos CSV para este fold
        train_csv_path = os.path.join(base_path, f"train_kfold_{fold+1}.csv")
        test_csv_path = os.path.join(base_path, f"test_kfold_{fold+1}.csv")
        
        # Guardar los DataFrames filtrados en archivos CSV
        train_data.to_csv(train_csv_path, index=False)
        test_data.to_csv(test_csv_path, index=False)
    
    return base_path

if __name__ == "__main__":
    
    trainparser = argparse.ArgumentParser(description='[StratifIAD] Parameters for training', allow_abbrev=False)
    trainparser.add_argument('-c','--config-file', type=str, default='/media/stefano/Seagate Expansion Drive/rochester_luis_emilio/LIMICM2_emilio/giancarlo_model/Classification/default_config_train_seg.yaml')
    args = trainparser.parse_args()
    conf = Dict(yaml.safe_load(open(args.config_file, "r")))

    k_fold_value=10

    #dir_dataset=create_files(k_fold_value,conf)
    #data_csv: 69 malign and benign videos
    #data_csv_new: 60 malign and benign videos
    dir_dataset='/media/stefano/Seagate Expansion Drive/rochester_luis_emilio/LIMICM2_emilio/giancarlo_model/prueba/data_csv_new'
 
    elementos = os.listdir(dir_dataset)
    archivos = [os.path.join(dir_dataset, elemento) for elemento in elementos]
    archivos_ordenados = sorted(archivos, key=lambda x: int(x.split("_")[-1].split(".")[0]))

    """
    dataset_test=archivos_ordenados[0]
    dataset_train=archivos_ordenados[1]
    print(dataset_test)
    print(dataset_train)
    trainer=train(dataset_train, dataset_test)
    wandb.finish()
    """

    for iteracion in range(0,k_fold_value*2,2):
        print(archivos_ordenados[iteracion])
        print(archivos_ordenados[iteracion+1])

    print("INICIO")
    for iteracion in range(0,k_fold_value*2,2):
        dataset_test=archivos_ordenados[iteracion]
        dataset_train=archivos_ordenados[iteracion+1]
        print(archivos_ordenados[iteracion])
        print(archivos_ordenados[iteracion+1])
        trainer=train(dataset_train, dataset_test,conf,k_fold_value)
        wandb.finish()
