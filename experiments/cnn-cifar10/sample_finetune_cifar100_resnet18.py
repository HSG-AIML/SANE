import os

# set environment variables to limit cpu usage
os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6


from SANE.datasets.dataset_tokens import DatasetTokens
from SANE.datasets.augmentations import (
    WindowCutter,
    PermutationAugmentation,
    #     CheckpointAugmentationPipeline,
)
from SANE.datasets.dataset_auxiliaries import (
    tokenize_checkpoint,
)
from SANE.models.def_AE_module import AEModule

import logging

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)


import json
import torch

from SANE.git_re_basin.git_re_basin import (
    PermutationSpec,
    resnet18_permutation_spec,
)

from pathlib import Path
from SANE.sampling.kde_sample_bootstrapped import sample_model_evaluation_bootstrapped
from SANE.evaluation.ray_fine_tuning_callback_subsampled import (
    CheckpointSamplingCallbackSubsampled,
)
from SANE.evaluation.ray_fine_tuning_callback import CheckpointSamplingCallback
from SANE.evaluation.ray_fine_tuning_callback_subsampled import (
    CheckpointSamplingCallbackSubsampled,
)
from SANE.evaluation.ray_fine_tuning_callback_bootstrapped import (
    CheckpointSamplingCallbackBootstrapped,
)


# %%
### cifar10 -> cifar10
# load reference model
from SANE.models.def_AE_module import AEModule

model_path = Path("path/to/your/model")

config = json.load(model_path.joinpath("params.json").open("r"))
# config["device"] = "cpu"
config["device"] = "cuda"
config["training::steps_per_epoch"] = 123
module = AEModule(config)

# load checkpoint - adjust the epoch to the one you want to evaluate
checkpoint = torch.load(
    model_path.joinpath("checkpoint_000100/state.pt"), map_location=config["device"]
)
module.model.load_state_dict(checkpoint["model"])
#

# subsampled tok halo bn_cond
clbk_10 = CheckpointSamplingCallbackBootstrapped(
    sample_config_path=Path("../../data/dataset_cnn_cifar10_sample_ep21/params.json"),
    finetuning_epochs=10,
    repetitions=10,
    tokensize=config["ae:i_dim"],
    anchor_ds_path=None,  # sample from distribution
    reference_dataset_path=Path("../../data/cifar10_preprocessed.pt"),
    bootstrap_iterations=20,
    bootstrap_samples=1000,
    bootstrap_keep_top_n=100,
    mode="token",
    norm_mode="standardize",
    layer_norms_path=Path(
        "../../data/dataset_cnn_cifar10_sample_ep21/dataset_normalization_train.json"
    ),
    logging_prefix="eval_cifar10_subsampled_token",
    every_n_epochs=0,
    eval_iterations=[0],
    batch_size=0,
    reset_classifier=False,
    halo=True,
    halo_wse=24,
    halo_hs=8,
    bn_condition_iters=200,
    anchor_sample_number=1000,
    mu_glob=0.0,
    sigma_glob=2.0,
)
res_10 = clbk_10.on_validation_epoch_end(ae_model=module, iteration=0)

fname = "perf_finetune_cifar10.json"
with open(fname, "w") as f:
    json.dump(res_10, f)
