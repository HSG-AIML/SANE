import logging

logging.basicConfig(level=logging.DEBUG)


from ray.tune import Trainable
from ray.tune.utils import wait_for_gpu
import torch
import sys

import json

# print(f"sys path in experiment: {sys.path}")
from pathlib import Path

# import model_definitions
from SANE.models.def_AE_module import AEModule

from torch.utils.data import DataLoader

from ray.air.integrations.wandb import setup_wandb

import logging

from SANE.datasets.augmentations import (
    AugmentationPipeline,
    TwoViewSplit,
    WindowCutter,
    ErasingAugmentation,
    NoiseAugmentation,
    MultiWindowCutter,
    StackBatches,
    PermutationSelector,
)


experiment_root = Path(
    "/netscratch2/kschuerholt/code/SANE/experiments/02_representation_learning/01_test/tune/ae_resnet_ffcv_permutation_test_1"
)
config_path = Path(
    "/netscratch2/kschuerholt/code/SANE/experiments/02_representation_learning/01_test/tune/ae_resnet_ffcv_permutation_test_1/AE_trainable_b0ae4_00000_0_ae_d_model=512,ae_nhead=16,ae_num_layers=16,training_windowsize=1024_2023-05-12_16-54-42/params.json"
)
config = json.load(config_path.open("r"))


from ffcv.loader import Loader, OrderOption

# import new downstream module
from SANE.models.downstream_module_ffcv import DownstreamTaskLearner

# trainloader
batch_size = config["trainset::batchsize"]
num_workers = config.get("testloader::workers", 4)
ordering = OrderOption.QUASI_RANDOM
# Dataset ordering
path_trainset = str(config["dataset::dump"]) + ".train"
trainloader = Loader(
    path_trainset,
    batch_size=batch_size,
    num_workers=num_workers,
    order=ordering,
    drop_last=True,
    # pipelines=PIPELINES
    os_cache=False,
)
# trainloader
batch_size = config["trainset::batchsize"]
num_workers = config.get("testloader::workers", 4)
ordering = OrderOption.SEQUENTIAL
# Dataset ordering
path_testset = str(config["dataset::dump"]) + ".test"
testloader = Loader(
    path_testset,
    batch_size=batch_size,
    num_workers=num_workers,
    order=ordering,
    drop_last=True,
    # pipelines=PIPELINES
    os_cache=False,
)
# config
batch_size = config["trainset::batchsize"]
num_workers = config.get("testloader::workers", 4)
ordering = OrderOption.SEQUENTIAL
# Dataset ordering
path_valset = str(config["dataset::dump"]) + ".val"
valloader = Loader(
    path_valset,
    batch_size=batch_size,
    num_workers=num_workers,
    order=ordering,
    drop_last=True,
    # pipelines=PIPELINES
    os_cache=False,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"device: {device}")


print(f"test dataloader")
# test dataloaders
for idx, batch in enumerate(trainloader):
    print(f"{idx} - {[bdx.shape for bdx in batch]}")
    if idx > 2:
        break

# test dataloaders
for idx, batch in enumerate(valloader):
    print(f"{idx} - {[bdx.shape for bdx in batch]}")
    if idx > 2:
        break

# test dataloaders
for idx, batch in enumerate(testloader):
    print(f"{idx} - {[bdx.shape for bdx in batch]}")
    if idx > 2:
        break

print(f"test dataloaders on devices")

# test dataloaders
for idx, batch in enumerate(valloader):
    batch = (ddx.to(device) for ddx in batch)
    print(f"{idx} - {[bdx.shape for bdx in batch]}")
    if idx > 2:
        break

# test dataloaders
for idx, batch in enumerate(testloader):
    batch = (ddx.to(device) for ddx in batch)
    print(f"{idx} - {[bdx.shape for bdx in batch]}")
    if idx > 2:
        break

# test dataloaders
for idx, batch in enumerate(trainloader):
    batch = (ddx.to(device) for ddx in batch)
    print(f"{idx} - {[bdx.shape for bdx in batch]}")
    if idx > 2:
        break


print(f"test train trafo dataloaders on devices")

stack_1 = []
windowsize = config.get("training::windowsize", 15)
if config.get("trainset::add_noise_view_1", 0.0) > 0.0:
    stack_1.append(NoiseAugmentation(config.get("trainset::add_noise_view_1", 0.0)))
if config.get("trainset::erase_augment", None) is not None:
    stack_1.append(ErasingAugmentation(**config["trainset::erase_augment"]))
stack_2 = []
if config.get("trainset::add_noise_view_2", 0.0) > 0.0:
    stack_2.append(NoiseAugmentation(config.get("trainset::add_noise_view_2", 0.0)))
if config.get("trainset::erase_augment", None) is not None:
    stack_2.append(ErasingAugmentation(**config["trainset::erase_augment"]))

stack_train = []
if config.get("trainset::multi_windows", None):
    stack_train.append(StackBatches())
else:
    stack_train.append(WindowCutter(windowsize=windowsize))
# put train stack together
if config.get("training::permutation_number", 0) == 0:
    split_mode = "copy"
    view_1_canon = True
    view_2_canon = True
else:
    split_mode = "permutation"
    view_1_canon = config.get("training::view_1_canon", True)
    view_2_canon = config.get("training::view_2_canon", False)
stack_train.append(
    TwoViewSplit(
        stack_1=stack_1,
        stack_2=stack_2,
        mode=split_mode,
        view_1_canon=view_1_canon,
        view_2_canon=view_2_canon,
    ),
)

trafo_train = AugmentationPipeline(stack=stack_train)

# test dataloaders
for idx, batch in enumerate(trainloader):
    #     batch = (ddx.to(device) for ddx in batch)
    batch = [sdx.to(device) for sdx in batch]
    print(f"{idx} - {[bdx.shape for bdx in batch]}")
    batch2 = trafo_train(*batch)
    print(f"{idx} - {[bdx.shape for bdx in batch2]}")
    if idx > 2:
        break


# test AUGMENTATIONS
stack_1 = []
if config.get("testset::add_noise_view_1", 0.0) > 0.0:
    stack_1.append(NoiseAugmentation(config.get("testset::add_noise_view_1", 0.0)))
if config.get("testset::erase_augment", None) is not None:
    stack_1.append(ErasingAugmentation(**config["testset::erase_augment"]))
stack_2 = []
if config.get("testset::add_noise_view_2", 0.0) > 0.0:
    stack_2.append(NoiseAugmentation(config.get("testset::add_noise_view_2", 0.0)))
if config.get("testset::erase_augment", None) is not None:
    stack_2.append(ErasingAugmentation(**config["testset::erase_augment"]))

stack_test = []
if config.get("trainset::multi_windows", None):
    stack_test.append(StackBatches())
else:
    stack_test.append(WindowCutter(windowsize=windowsize))
# put together
if config.get("testing::permutation_number", 0) == 0:
    split_mode = "copy"
    view_1_canon = True
    view_2_canon = True
else:
    split_mode = "permutation"
    view_1_canon = config.get("testing::view_1_canon", True)
    view_2_canon = config.get("testing::view_2_canon", False)
stack_test.append(
    TwoViewSplit(
        stack_1=stack_1,
        stack_2=stack_2,
        mode=split_mode,
        view_1_canon=view_1_canon,
        view_2_canon=view_2_canon,
    ),
)

# TODO: pass through permutation / view_1/2 canonical
trafo_test = AugmentationPipeline(stack=stack_test)

print(f"test val dataloader with test trafo on devices")

# test dataloaders
for idx, batch in enumerate(valloader):
    batch = [sdx.to(device) for sdx in batch]
    print(f"{idx} - {[bdx.shape for bdx in batch]}")
    batch2 = trafo_test(*batch)
    print(f"{idx} - {[bdx.shape for bdx in batch2]}")
    if idx > 2:
        break

print(f"test test dataloader with test trafo on devices")

# test dataloaders
for idx, batch in enumerate(testloader):
    batch = [sdx.to(device) for sdx in batch]
    print(f"{idx} - {[bdx.shape for bdx in batch]}")
    batch2 = trafo_test(*batch)
    print(f"{idx} - {[bdx.shape for bdx in batch2]}")
    if idx > 2:
        break

print(f"all good")
