from pathlib import Path

import json


import sys

from SANE.datasets.dataset_ffcv import prepare_ffcv_dataset

from SANE.git_re_basin.git_re_basin import (
    resnet18_permutation_spec,
    zoo_cnn_large_permutation_spec,
    zoo_cnn_permutation_spec,
)
import torch

from SANE.datasets.dataset_properties import PropertyDataset

import logging

logging.basicConfig(level=logging.INFO)


# prepare data
def prep_data():
    dataset_target_path = Path("./dataset_cifar100_token_288_ep60_std")
    zoo_path = [Path("./cifar100_resnet18_kaiming_uniform_ep60_no_opt/").absolute()]
    epoch_list = [
        60,
    ]
    permutation_spec = resnet18_permutation_spec()
    map_to_canonical = True
    standardize = True
    ds_split = [0.7, 0.15, 0.15]
    max_samples = 200
    weight_threshold = float("inf")
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

    page_size = 2**27
    # splits = ["train"]
    # splits = ["val"]
    splits = ["test"]

    result_key_list = ["test_acc", "training_iteration", "ggap"]
    config_key_list = []
    property_keys = {
        "result_keys": result_key_list,
        "config_keys": config_key_list,
    }

    ## to filter models by accuracy, uncomment the following lines
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

    prepare_ffcv_dataset(
        dataset_target_path=dataset_target_path,
        zoo_path=zoo_path,
        epoch_list=epoch_list,
        permutation_spec=permutation_spec,
        map_to_canonical=map_to_canonical,
        standardize=standardize,
        ds_split=ds_split,
        max_samples=max_samples,
        weight_threshold=weight_threshold,
        property_keys=property_keys,
        # filter_fn=filter_top_q,
        num_threads=num_threads,
        shuffle_path=shuffle_path,
        windowsize=windowsize,
        supersample=supersample,
        precision=precision,
        splits=splits,
        ignore_bn=ignore_bn,
        tokensize=tokensize,
        permutation_number_train=permutation_number_train,
        permutations_per_sample_train=permutations_per_sample_train,
        permutation_number_test=permutation_number_test,
        permutations_per_sample_test=permutations_per_sample_test,
        page_size=page_size,
        drop_pt_dataset=True,
    )


def infer_acc_threshold(root, epoch_list, quantile=0.9):
    logging.info("Inferring accuracy threshold")
    result_key_list = ["test_acc", "training_iteration", "ggap"]
    config_key_list = []
    property_keys = {
        "result_keys": result_key_list,
        "config_keys": config_key_list,
    }
    ds = PropertyDataset(
        root,  # path from which to load the dataset
        epoch_lst=epoch_list,  # list of epochs to load
        train_val_test="train",  # determines whcih dataset split to use
        ds_split=[
            1.0,
            0.0,
            0.0,
        ],  # sets ration between [train, test] or [train, val, test]
        property_keys=property_keys,  # keys of properties to load
    )
    acc_threshold = torch.quantile(torch.tensor(ds.properties["test_acc"]), q=quantile)
    return acc_threshold.item()


if __name__ == "__main__":
    prep_data()
