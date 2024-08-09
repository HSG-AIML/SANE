import os

# set environment variables to limit cpu usage
# os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
# os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
# os.environ["MKL_NUM_THREADS"] = "4"  # export MKL_NUM_THREADS=6
# os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
# os.environ["NUMEXPR_NUM_THREADS"] = "4"  # export NUMEXPR_NUM_THREADS=6
os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = "1"

import sys
from pathlib import Path

import json

import ray
from ray import tune

# # from ray.tune.logger import DEFAULT_LOGGERS
# from ray.tune.logger import JsonLogger, CSVLogger
# from ray.tune.integration.wandb import WandbLogger
import argparse
import torch
from torchvision import datasets, transforms

from ptmz.models.def_NN_experiment import NN_tune_trainable

data_path = Path("../../data")


def main():

    cifar_path = data_path.joinpath("image_data/CIFAR100")
    train_transforms = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761],
            ),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761],
            ),
        ]
    )

    trainset = datasets.CIFAR100(
        root=cifar_path,
        train=True,
        transform=train_transforms,
        download=True,
    )

    testset = datasets.CIFAR100(
        root=cifar_path,
        train=False,
        transform=test_transforms,
        download=True,
    )

    # save dataset and seed in data directory
    dataset = {
        "trainset": trainset,
        "testset": testset,
    }
    torch.save(dataset, data_path.joinpath("cifar100_preprocessed.pt"))


if __name__ == "__main__":
    main()
