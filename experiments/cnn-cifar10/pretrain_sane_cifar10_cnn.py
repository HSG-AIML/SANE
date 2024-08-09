import logging

logging.basicConfig(level=logging.INFO)

import os

# set environment variables to limit cpu usage
os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6

import torch

import ray
from ray import tune

from ray.air.integrations.wandb import WandbLoggerCallback
from SANE.evaluation.ray_fine_tuning_callback import CheckpointSamplingCallback
from SANE.evaluation.ray_fine_tuning_callback_subsampled import (
    CheckpointSamplingCallbackSubsampled,
)
from SANE.evaluation.ray_fine_tuning_callback_bootstrapped import (
    CheckpointSamplingCallbackBootstrapped,
)

import json

from pathlib import Path


from SANE.models.def_AE_trainable import AE_trainable
from SANE.datasets.dataset_sampling_preprocessed import PreprocessedSamplingDataset


PATH_ROOT = Path("./")


def main():
    ### set experiment resources ####
    print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
    # ray init to limit memory and storage
    cpus_per_trial = 10
    gpus_per_trial = 1
    gpus = 1
    cpus = gpus * cpus_per_trial

    # round down to maximize GPU usage

    resources_per_trial = {"cpu": cpus_per_trial, "gpu": gpus_per_trial}
    print(f"resources_per_trial: {resources_per_trial}")

    ### configure experiment #########
    experiment_name = "sane_cifar10_resnet18"
    # set module parameters
    config = {}
    config["seed"] = 32
    config["device"] = "cuda"
    config["device_no"] = 1
    config["training::precision"] = "amp"
    config["trainset::batchsize"] = 32

    config["ae:transformer_type"] = "gpt2"
    config["model::compile"] = True

    # permutation specs
    config["training::permutation_number"] = 5
    config["training::view_2_canon"] = False
    config["training::view_2_canon"] = True
    config["testing::permutation_number"] = 5
    config["testing::view_1_canon"] = True
    config["testing::view_2_canon"] = False

    config["training::reduction"] = "mean"

    config["ae:i_dim"] = 288
    config["ae:lat_dim"] = 128
    config["ae:max_positions"] = [55000, 100, 550]
    config["training::windowsize"] = 256
    config["ae:d_model"] = 2048
    config["ae:nhead"] = 16
    config["ae:num_layers"] = 8

    # configure optimizer
    config["optim::optimizer"] = "adamw"
    config["optim::lr"] = 1e-4
    config["optim::wd"] = 3e-9
    config["optim::scheduler"] = "OneCycleLR"

    # training config
    config["training::temperature"] = 0.1
    config["training::gamma"] = 0.05
    config["training::reduction"] = "mean"
    config["training::contrast"] = "simclr"
    # AMP
    #
    config["training::epochs_train"] = 50
    config["training::output_epoch"] = 25
    config["training::test_epochs"] = 1

    config["monitor_memory"] = True

    # configure output path
    output_dir = PATH_ROOT.joinpath("sane_pretraining")
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        pass

    ###### Datasets ###########################################################################
    # pre-compute dataset and drop in torch.save
    # data_path = output_dir.joinpath(experiment_name)
    data_path = Path("../../data/dataset_cnn_cifar10_sample_ep21-25_std/")
    data_path.mkdir(exist_ok=True)
    # path to ffcv dataset for training
    config["dataset::dump"] = data_path.joinpath("dataset.pt").absolute()
    config["downstreamtask::dataset"] = None
    # call dataset prepper function
    logging.info("prepare data")
    # prep_data(target_dataset_path=data_path)

    ### Augmentations
    config["trainloader::workers"] = 6
    config["trainset::add_noise_view_1"] = 0.1
    config["trainset::add_noise_view_2"] = 0.1
    config["trainset::noise_multiplicative"] = True
    config["trainset::erase_augment_view_1"] = None
    config["trainset::erase_augment_view_2"] = None

    config["callbacks"] = []

    config["resources"] = resources_per_trial
    context = ray.init(
        num_cpus=cpus,
        num_gpus=gpus,
        include_dashboard=True,
        dashboard_host="0.0.0.0",  # 0.0.0.0 is the host of the docker, (localhost is the container) (https://github.com/ray-project/ray/issues/11457#issuecomment-1325344221)
        dashboard_port=8265,
    )
    assert ray.is_initialized() == True

    print(f"started ray. running dashboard under {context.dashboard_url}")

    experiment = ray.tune.Experiment(
        name=experiment_name,
        run=AE_trainable,
        stop={
            "training_iteration": config["training::epochs_train"],
        },
        checkpoint_config=ray.air.CheckpointConfig(
            num_to_keep=None,
            checkpoint_frequency=config["training::output_epoch"],
            checkpoint_at_end=True,
        ),
        config=config,
        local_dir=output_dir,
        resources_per_trial=resources_per_trial,
    )
    # run
    ray.tune.run_experiments(
        experiments=experiment,
        resume=False,  # resumes from previous run. if run should be done all over, set resume=False
        # resume=True,  # resumes from previous run. if run should be done all over, set resume=False
        reuse_actors=False,
        verbose=3,
    )

    ray.shutdown()
    assert ray.is_initialized() == False


if __name__ == "__main__":
    main()
