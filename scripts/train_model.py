"""
This script is for training a model on the tensorfood dataset.
"""

import logging
import os

import hydra
import mlflow
import omegaconf
import tensorflow as tf
from dotenv import load_dotenv
from omegaconf import DictConfig

from src.utils.image_utils import create_tf_datasets
from src.modelling.model_factory import ModelFactory
from src.modelling.model_trainer import ModelTrainer
from src.utils import general_utils

os.environ["CUDA_VISIBLE_DEVICES"] = ""


@hydra.main(version_base=None, config_path="../conf", config_name="train_model.yaml")
def main(cfg: DictConfig) -> None:
    """
    Main function to orchestrate model training and evaluation.
    """
    # Setup logging
    logger = logging.getLogger(__name__)
    logger.info("Setting up logging configuration.")
    general_utils.setup_logging(
        logging_config_path=os.path.join(
            hydra.utils.get_original_cwd(), "conf", "logging.yaml"
        )
    )

    # Setup mlflow
    load_dotenv()  # Load environment variables from .env file

    mlflow_init_status, mlflow_run = general_utils.mlflow_init(
        cfg["mlflow"],
        setup_mlflow=cfg.mlflow.setup_mlflow,
        autolog=cfg.mlflow.mlflow_autolog,
    )

    # Log the parameters and configurations
    general_utils.mlflow_log(
        mlflow_init_status,
        "log_params",
        params={
            "model_name": cfg.training.base_model,
            "epochs": cfg.training.epochs,
            "learning_rate": cfg.training.learning_rate,
            "batch_size": cfg.training.batch_size,
        },
    )
    # Log the configuration file
    general_utils.mlflow_log(
        mlflow_init_status,
        "log_dict",
        dictionary=omegaconf.OmegaConf.to_container(cfg, resolve=True),
        artifact_file="train_config.json",
    )

    tf.random.set_seed(cfg.random_seed)

    logger.info("Starting the training pipeline.")

    # Create TensorFlow datasets
    train_ds, val_ds, test_ds = create_tf_datasets(
        train_dir_path=cfg.training.train_dir,
        val_dir_path=cfg.training.val_dir,
        test_dir_path=cfg.training.test_dir,
        image_size=(cfg.preprocessing.img_height, cfg.preprocessing.img_width),
        batch_size=cfg.training.batch_size,
    )

    # Load the model
    model_factory = ModelFactory()
    model = model_factory.build_model(
        base_model_name=cfg.training.base_model,
        input_shape=(cfg.preprocessing.img_height, cfg.preprocessing.img_width, 3),
        num_classes=cfg.training.num_classes,
    )
    print(model.summary())

    # devices = tf.config.list_physical_devices()
    # print("\nDevices: ", devices)

    # gpus = tf.config.list_physical_devices("GPU")
    # if gpus:
    #     details = tf.config.experimental.get_device_details(gpus[0])
    #     print("GPU details: ", details)

    trainer = ModelTrainer(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        optimizer=cfg.training.optimizer,
        learning_rate=cfg.training.learning_rate,
        scoring_metrics=list(cfg.training.metrics),
        device=cfg.training.device,
    )

    # Train the model
    logger.info("Starting training.")
    history = trainer.fit(epochs=cfg.training.epochs, verbose=cfg.training.verbose)

    for key, value in history.history.items():
        for i, val in enumerate(value):
            general_utils.mlflow_log(
                mlflow_init_status,
                "log_metric",
                key=key,
                value=val,
                step=i,
            )

    # Evaluate the model
    results = trainer.evaluate(test_ds)
    for key, value in results.items():
        general_utils.mlflow_log(mlflow_init_status, "log_metric", key=key, value=value)

    if cfg.training.save_model:
        save_model_path = trainer.save(
            save_model_dir=cfg.training.save_model_dir,
            model_name=cfg.training.base_model,
        )
        general_utils.mlflow_log(
            mlflow_init_status,
            "log_artifact",
            local_path=save_model_path,
            artifact_path="model",
        )

    if mlflow_init_status:
        artifact_uri = mlflow.get_artifact_uri()
        logger.info("Artifact URI: %s", artifact_uri)
        general_utils.mlflow_log(
            mlflow_init_status, "log_params", params={"artifact_uri": artifact_uri}
        )
        logger.info(
            "Model training with MLflow run ID %s has completed.",
            mlflow_run.info.run_id,
        )
        mlflow.end_run()
    else:
        logger.info("Model training has completed.")

    # Outputs for conf/train_model.yaml for hydra.sweeper.direction
    val_loss, val_f1_score = (
        history.history["val_loss"][-1],  # minimum validation loss
        history.history["val_f1_score"][-1],  # maximum validation accuracy
    )
    return val_loss, val_f1_score


if __name__ == "__main__":
    main()
