import json
from typing import Union, List, Any, Optional
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

# Lit-Diffusion
from lit_diffusion.diffusion_base.lit_diffusion_base import LitDiffusionBase

# SANE
from SANE.sampling.ddpm_sample import sample_model_evaluation


class CheckpointFineTuningCallback(Callback):
    def __init__(
        self,
        sample_config_path: Union[str, Path],
        finetuning_epochs: int,
        repetitions: int,
        tokensize: int,
        norm_mode: str,
        layer_norms_path: Union[str, Path],
        every_n_epochs: int,
        properties: Optional[List[Any]] = None,
    ):
        sample_config_path = Path(sample_config_path)
        with sample_config_path.open("r") as sample_config_file:
            self.sample_config = json.load(sample_config_file)
        layer_norms_path = Path(layer_norms_path)
        self.finetuning_epochs = finetuning_epochs
        self.repetitions = repetitions
        self.tokensize = tokensize
        self.properties = properties

        self.norm_mode = norm_mode
        with layer_norms_path.open("r") as layer_norms_file:
            self.layer_norms = json.load(layer_norms_file)

        self.every_n_epochs = every_n_epochs

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if (
            trainer.current_epoch != 0
            and trainer.current_epoch % self.every_n_epochs == 0
        ):
            assert isinstance(
                pl_module, LitDiffusionBase
            ), f"{self.__class__.__name__} only supports lightning modules which implement {LitDiffusionBase.__class__.__name__}"
            self.sample_config["optim::scheduler"] = None
            metrics_dict = sample_model_evaluation(
                ddpm_model=pl_module,
                tokensize=self.tokensize,
                sample_config=self.sample_config,
                finetuning_epochs=self.finetuning_epochs,
                repetitions=self.repetitions,
                norm_mode=self.norm_mode,
                layer_norms=self.layer_norms,
                properties=self.properties,
            )
            logging_dict = {}
            for k, v_list in metrics_dict.items():
                for idx, value in enumerate(v_list):
                    logging_dict[f"{k}_epoch_{idx}"] = value
            pl_module.log_dict(logging_dict)
