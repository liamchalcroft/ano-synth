import logging
from typing import List, Optional, Union

import numpy as np
import torch

from pythae.customexception import DatasetError
from pythae.data.preprocessors import BaseDataset, DataProcessor
from pythae.models import BaseAE, BaseAEConfig
from pythae.trainers.base_trainer.base_training_config import BaseTrainerConfig
from pythae.trainers import *
from pythae.trainers.training_callbacks import TrainingCallback, wandb_is_available, rename_logs
from pythae.data.datasets import DatasetOutput
from pythae.pipelines.base_pipeline import Pipeline

logger = logging.getLogger(__name__)

# make it print to the console.
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class DummyDataset:
    def __init__(self):
        self.data = None

    def __getitem__(self, idx):
        return DatasetOutput(data=torch.randn(1,192,192).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')))

    def __len__(self):
        return 100


class TrainingPipeline(Pipeline):
    """
    This Pipeline provides an end to end way to train your VAE model.
    The trained model will be saved in ``output_dir`` stated in the
    :class:`~pythae.trainers.BaseTrainerConfig`. A folder
    ``training_YYYY-MM-DD_hh-mm-ss`` is
    created where checkpoints and final model will be saved. Checkpoints are saved in
    ``checkpoint_epoch_{epoch}`` folder (optimizer and training config
    saved as well to resume training if needed)
    and the final model is saved in a ``final_model`` folder. If ``output_dir`` is
    None, data is saved in ``dummy_output_dir/training_YYYY-MM-DD_hh-mm-ss`` is created.
    Parameters:
        model (Optional[BaseAE]): An instance of :class:`~pythae.models.BaseAE` you want to train.
            If None, a default :class:`~pythae.models.VAE` model is used. Default: None.
        training_config (Optional[BaseTrainerConfig]): An instance of
            :class:`~pythae.trainers.BaseTrainerConfig` stating the training
            parameters. If None, a default configuration is used.
    """

    def __init__(
        self,
        model: Optional[BaseAE],
        training_config: Optional[BaseTrainerConfig] = None,
    ):

        if training_config is None:
            if model.model_name == "RAE_L2":
                training_config = CoupledOptimizerTrainerConfig(
                    encoder_optim_decay=0,
                    decoder_optim_decay=model.model_config.reg_weight,
                )

            elif (
                model.model_name == "Adversarial_AE" or model.model_name == "FactorVAE"
            ):
                training_config = AdversarialTrainerConfig()

            elif model.model_name == "VAEGAN":
                training_config = CoupledOptimizerAdversarialTrainerConfig()

            else:
                training_config = BaseTrainerConfig()

        elif model.model_name == "RAE_L2" or model.model_name == "PIWAE":
            if not isinstance(training_config, CoupledOptimizerTrainerConfig):

                raise AssertionError(
                    "A 'CoupledOptimizerTrainerConfig' "
                    f"is expected for training a {model.model_name}"
                )
            if model.model_name == "RAE_L2":
                if training_config.decoder_optimizer_params is None:
                    training_config.decoder_optimizer_params = {
                        "weight_decay": model.model_config.reg_weight
                    }
                else:
                    training_config.decoder_optimizer_params[
                        "weight_decay"
                    ] = model.model_config.reg_weight

        elif model.model_name == "Adversarial_AE" or model.model_name == "FactorVAE":
            if not isinstance(training_config, AdversarialTrainerConfig):

                raise AssertionError(
                    "A 'AdversarialTrainer' "
                    f"is expected for training a {model.model_name}"
                )

        elif model.model_name == "VAEGAN":
            if not isinstance(
                training_config, CoupledOptimizerAdversarialTrainerConfig
            ):

                raise AssertionError(
                    "A 'CoupledOptimizerAdversarialTrainer' "
                    "is expected for training a VAEGAN"
                )

        if not isinstance(training_config, BaseTrainerConfig):
            raise AssertionError(
                "A 'BaseTrainerConfig' " "is expected for the pipeline"
            )

        self.data_processor = DataProcessor()
        self.model = model
        self.training_config = training_config

    def _check_dataset(self, dataset: BaseDataset):

        try:
            dataset_output = dataset[0]

        except Exception as e:
            raise DatasetError(
                "Error when trying to collect data from the dataset. Check `__getitem__` method. "
                "The Dataset should output a dictionnary with keys at least ['data']. "
                "Please check documentation.\n"
                f"Exception raised: {type(e)} with message: " + str(e)
            ) from e

        if "data" not in dataset_output.keys():
            raise DatasetError(
                "The Dataset should output a dictionnary with keys ['data']"
            )

        try:
            len(dataset)

        except Exception as e:
            raise DatasetError(
                "Error when trying to get dataset len. Check `__len__` method. "
                "Please check documentation.\n"
                f"Exception raised: {type(e)} with message: " + str(e)
            ) from e

        # check everything if fine when combined with data loader
        from torch.utils.data import DataLoader

        dataloader = DataLoader(dataset=dataset, batch_size=min(len(dataset), 2))
        loader_out = next(iter(dataloader))
        assert loader_out['data'].shape[0] == min(
            len(dataset), 2
        ), "Error when combining dataset with loader."

    def __call__(
        self,
        train_data: Union[np.ndarray, torch.Tensor, torch.utils.data.Dataset],
        eval_data: Union[np.ndarray, torch.Tensor, torch.utils.data.Dataset] = None,
        callbacks: List[TrainingCallback] = None,
        epoch: int = 1,
        optimizer_state_dict = None,
        scheduler_state_dict = None,
        ffcv_train = None,
        ffcv_val = None
    ):
        """
        Launch the model training on the provided data.
        Args:
            training_data (Union[~numpy.ndarray, ~torch.Tensor]): The training data as a
                :class:`numpy.ndarray` or :class:`torch.Tensor` of shape (mini_batch x
                n_channels x ...)
            eval_data (Optional[Union[~numpy.ndarray, ~torch.Tensor]]): The evaluation data as a
                :class:`numpy.ndarray` or :class:`torch.Tensor` of shape (mini_batch x
                n_channels x ...). If None, only uses train_fata for training. Default: None.
            callbacks (List[~pythae.trainers.training_callbacks.TrainingCallbacks]):
                A list of callbacks to use during training.
        """

        if isinstance(train_data, np.ndarray) or isinstance(train_data, torch.Tensor):

            logger.info("Preprocessing train data...")
            train_data = self.data_processor.process_data(train_data)
            train_dataset = self.data_processor.to_dataset(train_data)

        else:
            train_dataset = train_data

        if ffcv_train is None:
            logger.info("Checking train dataset...")
            self._check_dataset(train_dataset)
        else:
            train_dataset = DummyDataset()

        if eval_data is not None:
            if isinstance(eval_data, np.ndarray) or isinstance(eval_data, torch.Tensor):
                logger.info("Preprocessing eval data...\n")
                eval_data = self.data_processor.process_data(eval_data)
                eval_dataset = self.data_processor.to_dataset(eval_data)

            else:
                eval_dataset = eval_data

            if ffcv_val is None:
                logger.info("Checking eval dataset...")
                self._check_dataset(eval_dataset)
            else:
                eval_dataset = DummyDataset()

        else:
            eval_dataset = None

        if isinstance(self.training_config, CoupledOptimizerTrainerConfig):
            logger.info("Using Coupled Optimizer Trainer\n")
            trainer = CoupledOptimizerTrainer(
                model=self.model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                training_config=self.training_config,
                callbacks=callbacks,
            )

        elif isinstance(self.training_config, AdversarialTrainerConfig):
            logger.info("Using Adversarial Trainer\n")
            trainer = AdversarialTrainer(
                model=self.model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                training_config=self.training_config,
                callbacks=callbacks,
            )

        elif isinstance(self.training_config, CoupledOptimizerAdversarialTrainerConfig):
            logger.info("Using Coupled Optimizer Adversarial Trainer\n")
            trainer = CoupledOptimizerAdversarialTrainer(
                model=self.model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                training_config=self.training_config,
                callbacks=callbacks,
            )

        elif isinstance(self.training_config, BaseTrainerConfig):
            logger.info("Using Base Trainer\n")
            trainer = BaseTrainer(
                model=self.model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                training_config=self.training_config,
                callbacks=callbacks,
                ffcv_device=(ffcv_train is not None),
            )

        self.trainer = trainer
        if optimizer_state_dict is not None:
            self.trainer.optimizer.load_state_dict(optimizer_state_dict)
        if scheduler_state_dict is not None:
            self.trainer.scheduler.load_state_dict(scheduler_state_dict)

        if ffcv_train is not None:
            self.trainer.train_loader = ffcv_train
        if ffcv_val is not None:
            self.trainer.val_loader = ffcv_val

        trainer.train(start_epoch=epoch)


class WandbCallback(TrainingCallback):  # pragma: no cover
    """
    A :class:`TrainingCallback` integrating the experiment tracking tool
    `wandb` (https://wandb.ai/).
    It allows users to store their configs, monitor their trainings
    and compare runs through a graphic interface. To be able use this feature you will need:
        - a valid `wandb` account
        - the package `wandb` installed in your virtual env. If not you can install it with
        .. code-block::
            $ pip install wandb
        - to be logged in to your wandb account using
        .. code-block::
            $ wandb login
    """

    def __init__(self):
        if not wandb_is_available():
            raise ModuleNotFoundError(
                "`wandb` package must be installed. Run `pip install wandb`"
            )

        else:
            import wandb

            self._wandb = wandb

    def setup(
        self,
        training_config: BaseTrainerConfig,
        model_config: BaseAEConfig = None,
        project_name: str = "pythae_experiment",
        entity_name: str = None,
        name: str = None,
        **kwargs,
    ):
        """
        Setup the WandbCallback.
        args:
            training_config (BaseTrainerConfig): The training configuration used in the run.
            model_config (BaseAEConfig): The model configuration used in the run.
            project_name (str): The name of the wandb project to use.
            entity_name (str): The name of the wandb entity to use.
        """

        self.is_initialized = True

        training_config_dict = training_config.to_dict()

        self.run = self._wandb.init(project=project_name, entity=entity_name, name=name)

        if model_config is not None:
            model_config_dict = model_config.to_dict()

            self._wandb.config.update(
                {
                    "training_config": training_config_dict,
                    "model_config": model_config_dict,
                }
            )

        else:
            self._wandb.config.update({**training_config_dict})

        self._wandb.define_metric("train/global_step")
        self._wandb.define_metric("*", step_metric="train/global_step", step_sync=True)

    def on_train_begin(self, training_config: BaseTrainerConfig, **kwargs):
        model_config = kwargs.pop("model_config", None)
        if not self.is_initialized:
            self.setup(training_config, model_config=model_config)

    def on_log(self, training_config: BaseTrainerConfig, logs, **kwargs):
        global_step = kwargs.pop("global_step", None)
        logs = rename_logs(logs)

        self._wandb.log({**logs, "train/global_step": global_step})

    def on_prediction_step(self, training_config: BaseTrainerConfig, **kwargs):
        kwargs.pop("global_step", None)

        column_names = ["images_id", "truth", "reconstruction", "normal_generation"]

        true_data = kwargs.pop("true_data", None)
        reconstructions = kwargs.pop("reconstructions", None)
        generations = kwargs.pop("generations", None)

        data_to_log = []

        if (
            true_data is not None
            and reconstructions is not None
            and generations is not None
        ):
            for i in range(len(true_data)):

                data_to_log.append(
                    [
                        f"img_{i}",
                        self._wandb.Image(
                            np.moveaxis(true_data[i].cpu().detach().numpy() * 255, 0, -1)
                        ),
                        self._wandb.Image(
                            np.clip(
                                np.moveaxis(
                                    reconstructions[i].cpu().detach().numpy() * 255, 0, -1
                                ),
                                0,
                                255.0,
                            )
                        ),
                        self._wandb.Image(
                            np.clip(
                                np.moveaxis(
                                    generations[i].cpu().detach().numpy() * 255, 0, -1
                                ),
                                0,
                                255.0,
                            )
                        ),
                    ]
                )

            val_table = self._wandb.Table(data=data_to_log, columns=column_names)

            self._wandb.log({"my_val_table": val_table})

    def on_train_end(self, training_config: BaseTrainerConfig, **kwargs):
        self.run.finish()
