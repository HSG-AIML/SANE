# prepare data
from pathlib import Path

from SANE.git_re_basin.git_re_basin import (
    resnet18_permutation_spec,
)

from SANE.datasets.dataset_preprocessing import prepare_multiple_datasets
from SANE.datasets.dataset_properties import PropertyDataset
from SANE.datasets.dataset_sampling_preprocessed import PreprocessedSamplingDataset

import logging
import torch

logging.basicConfig(level=logging.INFO)

## to filter models by accuracy, uncomment the following lines and add the filter fn (filter_top_q) to the prep_data function

# def infer_acc_threshold(root, epoch_list, quantile=0.9):
#     logging.info("Inferring accuracy threshold")
#     result_key_list = ["test_acc", "training_iteration", "ggap"]
#     config_key_list = []
#     property_keys = {
#         "result_keys": result_key_list,
#         "config_keys": config_key_list,
#     }
#     ds = PropertyDataset(
#         root,  # path from which to load the dataset
#         epoch_lst=epoch_list,  # list of epochs to load
#         train_val_test="train",  # determines whcih dataset split to use
#         ds_split=[
#             1.0,
#             0.0,
#             0.0,
#         ],  # sets ration between [train, test] or [train, val, test]
#         property_keys=property_keys,  # keys of properties to load
#     )
#     acc_threshold = torch.quantile(torch.tensor(ds.properties["test_acc"]), q=quantile)
#     return acc_threshold.item()

# acc_thresh = infer_acc_threshold(zoo_path, epoch_list, quantile=0.70)

# def filter_top_q(path):
#     fname = Path(path).joinpath("result.json")
#     restmp = {}
#     for line in fname.open():
#         restmp = json.loads(line)
#         if restmp["training_iteration"] in epoch_list:
#             break
#         else:
#             restmp = {}
#     if restmp.get("test_acc", 0.0) > acc_thresh:
#         return False
#     else:
#         return True


def prep_data():
    dataset_target_path = [
        Path("./dataset_resnet18_cifar10_token_288_ep21-25_std/"),
    ]
    zoo_path = [  
        Path("./cifar10_resnet18_kaiming_uniform/").absolute()
    ]
    zoo_path_and_permutation_spec_and_target_path = [
        (zoo_path[0], resnet18_permutation_spec, dataset_target_path[0]),
    ]
    configurations = create_configurations(zoo_path_and_permutation_spec_and_target_path, filter_fn=None)
    prepare_multiple_datasets(configurations=configurations)

    # create dataset dump for later use
    ds_train = PreprocessedSamplingDataset(zoo_paths=dataset_target_path, split="train")
    ds_val = PreprocessedSamplingDataset(zoo_paths=dataset_target_path, split="val")
    ds_test = PreprocessedSamplingDataset(zoo_paths=dataset_target_path, split="test")

    dataset = {
        "trainset": ds_train,
        "valset": ds_val,
        "testset": ds_test,
    }

    torch.save(dataset, dataset_target_path[0] / "dataset.pt")


def create_configurations(zoo_path_and_permutation_spec_and_target_path, filter_fn=None):
    # static parameters
    epoch_list = [21, 22, 23, 24, 25]
    map_to_canonical = True
    standardize = True
    ds_split = [0.7, 0.15, 0.15]
    max_samples = 200
    weight_threshold = 15000
    num_threads = 5
    shuffle_path = True
    windowsize = 2048
    supersample = 50
    precision = "32"
    ignore_bn = True
    tokensize = 288

    # permutation spec
    permutation_number_train = 200
    permutations_per_sample_train = 5
    permutation_number_test = 10
    permutations_per_sample_test = 5

    # dataset splits
    splits = ["train", "val", "test"]

    result_key_list = ["test_acc", "training_iteration", "ggap"]
    config_key_list = ["model::type"]
    property_keys = {
        "result_keys": result_key_list,
        "config_keys": config_key_list,
    }

    configurations = []
    for split in splits:
        # dynamic parameters
        for zoo_path, permutation_spec, dataset_target_path in zoo_path_and_permutation_spec_and_target_path:
            configurations.append(
                {
                    "dataset_target_path": dataset_target_path,
                    "zoo_path": zoo_path,
                    "epoch_list": epoch_list,
                    "permutation_spec": permutation_spec,
                    "map_to_canonical": map_to_canonical,
                    "standardize": standardize,
                    "ds_split": ds_split,
                    "max_samples": max_samples,
                    "weight_threshold": weight_threshold,
                    "num_threads": num_threads,
                    "shuffle_path": shuffle_path,
                    "windowsize": windowsize,
                    "supersample": supersample,
                    "precision": precision,
                    "ignore_bn": ignore_bn,
                    "tokensize": tokensize,
                    "permutation_number_train": permutation_number_train,
                    "permutations_per_sample_train": permutations_per_sample_train,
                    "permutation_number_test": permutation_number_test,
                    "permutations_per_sample_test": permutations_per_sample_test,
                    "splits": [split],
                    "property_keys": property_keys,
                    "drop_pt_dataset": False,
                    "filter_fn": filter_fn,
                }
            )
    
    return configurations


if __name__ == "__main__":
    prep_data()