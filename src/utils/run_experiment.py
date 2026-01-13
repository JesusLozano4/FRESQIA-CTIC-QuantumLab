import numpy as np
import torch
import time
from loguru import logger
import mlflow
import os

from src.utils.dataset import load_dataset
from src.nn.models.quantum.QCNN import QuantumCircuitModel
from src.nn.models.hybrid.HQNN_quanv import FlexHybridCNN
from src.utils.training import Trainer

def run_experiment(
    dataset_cfg,
    embedding_cfg,
    circuit_cfg,
    measurement_cfg,
    hparams,
    wshape,
    reshape_cfg,
    cont,
    save,
):
    """
    Prepare data, create model, trainer, and run training for one combination of config.
    """
    # Unpack dataset settings
    dataset_name = dataset_cfg["dataset_name"]
    limit = dataset_cfg["limit"]
    image_size = dataset_cfg["image_size"]
    test_size = dataset_cfg["test_size"]
    output = dataset_cfg["output"]
    allowed_classes = dataset_cfg["allowed_classes"]
    n_classes = len(allowed_classes)
    n_wires = np.ceil(np.log2(3*16**2)).astype(int)
    weight_shapes = wshape

    # Unpack hyperparameters
    epochs = hparams["epochs"]
    lr = hparams["learning_rate"]
    early_stopping = hparams["early_stopping"]
    patience = hparams["patience"]
    use_schedulefree = hparams["use_schedulefree"]
    use_quantum = hparams["use_quantum"]
    plot = hparams["plot"]
    log_mlflow = hparams["log_mlflow"]


    # Loguru info: Start of run
    logger.info(f"Starting run: dataset={dataset_name}, "
            f"embedding={embedding_cfg['name']}, "
            f"circuit={circuit_cfg['name']}, measurement={measurement_cfg['name']}, "
            f"epochs={epochs}, lr={lr}")

    run_name = (
        f"QCNN_{dataset_name}_{image_size}x{image_size}_"
        f"emb={embedding_cfg['name']}_circuit={circuit_cfg['name']}_meas={measurement_cfg['name']}_"
        f"_lr={lr}_ep={epochs}_config{cont}"
    )
    # Create a dictionary of all configurations for MLflow
    mlflow_params = {
        # Dataset parameters
        "dataset_name": dataset_name,
        "limit": limit,
        "image_size": image_size,
        "test_size": test_size,
        "allowed_classes": str(allowed_classes),  # Convert list to string
        
        # Embedding parameters
        "embedding_name": embedding_cfg['name'],
        
        # Circuit parameters
        "circuit_name": circuit_cfg['name'],
        
        # Measurement parameters
        "measurement_name": measurement_cfg['name'],
        
        # Any other relevant parameters you want to track
        "run_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    embedding_params={
        "func": embedding_cfg["func"],
        "func_params": embedding_cfg["func_params"]
    }

    variational_params={
            "func": circuit_cfg["func"],
            "func_params": circuit_cfg["func_params"]  # includes 'weights' re-init
        }
    measurement_params={
        "func": measurement_cfg["func"],
        "func_params": measurement_cfg["func_params"]
    }

    
    
        
    mlflow_project_name = f"QCNN {dataset_name} {image_size}x{image_size}_random"
    
    # 1. Load Dataset
    train_loader, val_loader = load_dataset(
        dataset_name,
        output,
        limit,
        allowed_classes,
        image_size,
        test_size,
    )

    # 2. Create model
    if cont == "1a":
        model = QuantumCircuitModel(n_wires=n_wires,
            embedding=embedding_params,
            circuit=variational_params,
            measurement=measurement_params,
            weight_shapes=weight_shapes,
            reshaper=reshape_cfg
        )
        name = "QCNN"
    else:
        model = FlexHybridCNN(embedding_params = embedding_params,
                                    variational_params= variational_params,
                                    measurement_params= measurement_params,
                                    n_classes = n_classes,
                                    use_quantum = use_quantum,
                                    qkernel_shape = 3,
                                    epochs = epochs,
                                    dataset = "2hojas",
                                    image_size = image_size,
                                )
        name = "HybridCNN"

    # 3. Create Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        early_stopping=early_stopping,
        patience=patience,
        log=log_mlflow,
        mlflow_project=mlflow_project_name,
        mlflow_run_name=run_name,
        use_quantum=use_quantum,
        plot=plot,
        allowed_classes=allowed_classes,
        lr=lr,
        use_schedulefree=use_schedulefree,
        mlflow_params=mlflow_params,
    )
    logger.debug(f"Trainer created: early_stopping={early_stopping}, "
                 f"patience={patience}, log_mlflow={log_mlflow}")

    # 4. Train
    trainer.fit()
    if save:
        os.chdir("..")
        torch.save(model.state_dict(), os.getcwd() + f"/dataset/{dataset_name}_{name}.pth")
    logger.info(f"Finished run: {run_name}")