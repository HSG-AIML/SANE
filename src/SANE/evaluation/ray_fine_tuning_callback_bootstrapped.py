import json
from typing import Union, List, Any, Optional
from pathlib import Path

from ray.tune import Callback

# SANE
from SANE.sampling.kde_sample_bootstrapped import sample_model_evaluation_bootstrapped

from SANE.models.def_AE_module import AEModule

import torch


class CheckpointSamplingCallbackBootstrapped(Callback):
    def __init__(
        self,
        sample_config_path: Union[str, Path],
        finetuning_epochs: int,
        repetitions: int,
        anchor_ds_path: str,  # Path to anchor dataset
        reference_dataset_path: str,  # Path to reference dataset
        bootstrap_iterations: int,  # how many bootstrap iterations ()
        bootstrap_samples: int,  # how many samples to draw at bootstrapping
        bootstrap_keep_top_n,  # out of those samples, how many to keep for next round of bootstrapping?
        mode: str,  # 'individual','token,'joint'
        norm_mode: str,  # "standardize",etc
        layer_norms_path: Union[str, Path],
        logging_prefix: str = "eval",
        every_n_epochs: int = 5,
        eval_iterations: List[int] = [],
        batch_size: int = 0,
        reset_classifier: bool = False,
        halo: bool = False,
        halo_wse: int = 156,
        halo_hs: int = 64,
        bn_condition_iters: int = 0,
        mu_glob: float = 0.0,
        sigma_glob: float = 10.0,
        ensemble: bool = False,
        anchor_sample_number: int = 0,
        drop_samples_to_path: Optional[str | Path] = None,
    ):
        """
        Args:
            sample_config_path: Path to model config fine-tuning task
            finetuning_epochs: Number of fine-tuning epochs
            repetitions: Number of repetitions for fine-tuning models
            anchor_ds_path: Path to anchor dataset, which is used to fit the kde distribution to
            reference_dataset_path: path to reference (image) dataset
            mode: kde fitting mode to embeddings: 'individual','token,'joint'
            norm_mode: Normalization mode for embeddings: "standardize",etc
            layer_norms_path: Path to layer norms
            logging_prefix: Prefix for logging
            every_n_epochs: Evaluate every n epochs
            eval_iterations: List[int] itertions at which to evaluate
            batch_size: Batch size for embeedding anchor dataset
            reset_classifier: Reset classifier for fine-tuning
            halo (bool, optional): use halo-windows for encoding / decoding, instead of passing the entire sequence in one go. Defaults to False.
            halo_wse (int, optional): size of haloed-window. Defaults to 156.
            halo_hs (int, optional): size of the halo around the window. Defaults to 64.
            bn_condition_iters: (int, optional): if nonzero, perform conditioning iterations on train/val image dataset to tune bn statistics (only stats, no weight udpates)
            mu_glob: global mean for random init anchor samples
            sigma_glob: global variance for random init anchor samples
            ensemble: whether to use ensemble of models for sampling
            anchor_sample_number: number of anchor samples to use for sampling
        """
        super(CheckpointSamplingCallbackBootstrapped, self).__init__()

        sample_config_path = Path(sample_config_path)
        with sample_config_path.open("r") as sample_config_file:
            self.sample_config = json.load(sample_config_file)
        layer_norms_path = Path(layer_norms_path)
        self.finetuning_epochs = finetuning_epochs
        self.repetitions = repetitions

        self.anchor_ds_path = anchor_ds_path
        self.mode = mode

        self.norm_mode = norm_mode
        with layer_norms_path.open("r") as layer_norms_file:
            self.layer_norms = json.load(layer_norms_file)

        self.logging_prefix = logging_prefix

        self.every_n_epochs = every_n_epochs
        self.eval_iterations = eval_iterations
        if not len(self.eval_iterations) == 0 and self.every_n_epochs != 0:
            raise ValueError(
                "If eval_iterations is not empty, every_n_epochs must be 0"
            )
        elif len(self.eval_iterations) == 0:
            # infer eval iterations from every_n_epochs
            # assuming max 5000 epochs
            self.eval_iterations = list(range(0, 5000, self.every_n_epochs))

        self.batch_size = batch_size

        self.reference_dataset_path = reference_dataset_path

        self.bootstrap_iterations = bootstrap_iterations
        self.bootstrap_samples = bootstrap_samples
        self.bootstrap_keep_top_n = bootstrap_keep_top_n

        self.reset_classifier = reset_classifier

        self.halo = halo
        self.halo_wse = halo_wse
        self.halo_hs = halo_hs

        self.bn_condition_iters = bn_condition_iters

        self.mu_glob = mu_glob
        self.sigma_glob = sigma_glob

        self.ensemble = ensemble

        self.anchor_sample_number = anchor_sample_number

        self.drop_samples_to_path = drop_samples_to_path

    def on_validation_epoch_end(self, ae_model, iteration) -> None:
        results = {}

        if iteration > max(self.eval_iterations):
            # extend eval_iterations
            self.eval_iterations.extend(
                list(
                    range(
                        max(self.eval_iterations), iteration + 5000, self.every_n_epochs
                    )
                )
            )

        if iteration not in self.eval_iterations:
            return results

        # call sampling eval function
        metrics_dict = sample_model_evaluation_bootstrapped(
            ae_model=ae_model,
            sample_config=self.sample_config,
            finetuning_epochs=self.finetuning_epochs,
            repetitions=self.repetitions,
            anchor_ds_path=self.anchor_ds_path,
            reference_dataset_path=self.reference_dataset_path,
            bootstrap_iterations=self.bootstrap_iterations,
            bootstrap_samples=self.bootstrap_samples,
            bootstrap_keep_top_n=self.bootstrap_keep_top_n,
            mode=self.mode,
            norm_mode=self.norm_mode,
            layer_norms=self.layer_norms,
            batch_size=self.batch_size,
            reset_classifier=self.reset_classifier,
            halo=self.halo,
            halo_wse=self.halo_wse,
            halo_hs=self.halo_hs,
            bn_condition_iters=self.bn_condition_iters,
            mu_glob=self.mu_glob,
            sigma_glob=self.sigma_glob,
            ensemble=self.ensemble,
            anchor_sample_number=self.anchor_sample_number,
            drop_samples_to_path=self.drop_samples_to_path,
        )
        # Add the metric to the trial result dict
        for k, v_list in metrics_dict.items():
            # if list -> interpret as performance over epochs
            if isinstance(v_list, list):
                for idx, value in enumerate(v_list):
                    results[f"{self.logging_prefix}/{k}_epoch_{idx}"] = value
            # else: interpret as single value
            else:
                results[f"{self.logging_prefix}/{k}"] = v_list
        return results
