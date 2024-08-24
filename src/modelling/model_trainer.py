import logging
import os
from typing import List
from typing import Optional
from typing import Union

import tensorflow as tf
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import History
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.losses import Loss
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import SGD

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    A class to train a TensorFlow Keras model using specified training and validation datasets.

    Attributes:
        model (tf.keras.models.Model): The Keras model to be trained.
        train_ds (tf.data.Dataset): The training dataset.
        val_ds (Optional[tf.data.Dataset]): The validation dataset, if provided.
        optimizer (tf.keras.optimizers.Optimizer): The optimizer for training the model.
        learning_rate (float): The learning rate for the optimizer.
        loss_fn (tf.keras.losses.Loss): The loss function for training the model.
        scoring_metrics (List[str]): The metrics to be evaluated by the model during training and testing.

    Methods:
        _init_optimizer(optimizer: str, learning_rate: float) -> tf.keras.optimizers.Optimizer:
            Initializes the optimizer based on the provided name and learning rate.
        compile_model() -> None:
            Compiles the model with the specified optimizer, loss function, and metrics.
        train(epochs: int, callbacks: Optional[List[tf.keras.callbacks.Callback]] = None) -> History:
            Trains the model using the training and validation datasets.
    """

    def __init__(
        self,
        model: tf.keras.models.Model,
        train_ds: tf.data.Dataset,
        val_ds: Optional[tf.data.Dataset] = None,
        optimizer: str = "adam",
        learning_rate: float = 0.001,
        loss_fn: Loss = CategoricalCrossentropy(),
        scoring_metrics: List[str] = ["accuracy"],
        device: str = "/cpu:0",
    ):
        """
        Initializes the ModelTrainer with the specified model and training/validation datasets.

        Parameters:
        - model: A Keras Model instance to be trained.
        - train_ds: A tf.data.Dataset instance for training data.
        - val_ds: An optional tf.data.Dataset instance for validation data.
        - optimizer: A string specifying the optimizer to use ('adam' or 'sgd').
        - learning_rate: A float specifying the learning rate for the optimizer.
        - loss_fn: A Keras Loss instance or string specifying the loss function.
        - metrics: A list of strings specifying the metrics to be evaluated by the model during training and testing.
        """
        self.model = model
        self.device = device
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.optimizer = self._init_optimizer(optimizer, learning_rate)
        self.loss_fn = loss_fn
        self.metrics = scoring_metrics

    def _init_optimizer(
        self, optimizer: str, learning_rate: float
    ) -> Union[Adam, SGD, RMSprop, Adagrad, Adamax]:
        """
        Initializes the optimizer based on the provided arguments.

        Parameters:
        - optimizer: A string specifying the optimizer to use ('adam', 'sgd', 'rmsprop', 'adagrad', 'adamax').
        - learning_rate: A float specifying the learning rate for the optimizer.

        Returns:
        An instance of the specified optimizer with the given learning rate.
        """
        optimizer_functions = {
            "adam": Adam(learning_rate),
            "sgd": SGD(learning_rate),
            "rmsprop": RMSprop(learning_rate),
            "adagrad": Adagrad(learning_rate),
            "adamax": Adamax(learning_rate),
        }

        optimizer = optimizer.lower()
        if optimizer in optimizer_functions:
            return optimizer_functions[optimizer]
        else:
            raise ValueError(f"Optimizer {optimizer} not supported.")

    def fit(
        self,
        epochs: int = 100,
        batch_size: int = 32,
        callbacks: Optional[List[EarlyStopping]] = None,
        verbose: int = 1,
    ) -> History:
        """
        Trains the model using the training dataset and evaluates it using the validation dataset.

        Parameters:
        - epochs: An integer specifying the number of epochs to train the model.
        - batch_size: An integer specifying the number of samples per gradient update.
        - callbacks: A list of Keras Callback instances to apply during training.
        - verbose: An integer specifying the verbosity mode.

        Returns:
        A Keras History object containing the training history metrics.
        """
        self.model.compile(
            optimizer=self.optimizer, loss=self.loss_fn, metrics=self.metrics
        )
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_fn,
            metrics=[
                "accuracy",
                "precision",
                "recall",
                metrics.F1Score(average="macro", name="f1_score"),
            ],
        )
        with tf.device(self.device):
            history = self.model.fit(
                self.train_ds,
                validation_data=self.val_ds,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=verbose,
            )
            logger.info("Model training complete.")
            return history

    def evaluate(self, test_ds: tf.data.Dataset) -> dict:
        """
        Evaluates the model on the test dataset.

        Parameters:
        - test_ds: A tf.data.Dataset instance for testing data.

        Returns:
        A dictionary containing the evaluation results.
        """
        with tf.device(self.device):
            evaluation_results = self.model.evaluate(test_ds, return_dict=True)
            logger.info(f"Model evaluation results: {evaluation_results}")
            return evaluation_results

    def save(self, save_model_dir: str, model_name: str) -> str:
        """
        Saves the model to the specified file path.

        Parameters:
        - save_file_path: A string specifying the file path to save the model.

        Returns:
        A string containing the file path where the model was saved.
        """
        if not os.path.exists(save_model_dir):
            os.makedirs(save_model_dir)
        save_model_path = os.path.join(
            save_model_dir,
            model_name + ".keras",
        )
        self.model.save(save_model_path)
        logger.info(f"Model saved to {save_model_path}")
        return save_model_path
