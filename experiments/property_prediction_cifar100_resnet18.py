from SANE.models.def_downstream_module import (
    DownstreamTaskLearner as DownstreamTaskLearner,
)

from SANE.models.downstream_baselines import IdentityModel, LayerQuintiles
from SANE.datasets.dataset_tokens import DatasetTokens

from pathlib import Path

import json
import os
import torch

import logging

logging.basicConfig(level=logging.INFO)
from SANE.git_re_basin.git_re_basin import (
    resnet18_permutation_spec,
    zoo_cnn_large_permutation_spec,
    zoo_cnn_permutation_spec,
)
from SANE.models.def_AE_module import AEModule


# %%
logging.info("Loading config")
results_json_path = "cifar_100_resnet.json"

# load reference model


model_tag = "cifar100_resnet18_downstreamtask"

model_path = Path("path/to/your/model")
config = json.load(model_path.joinpath("params.json").open("r"))
device = "cuda" if torch.cuda.is_available() else "cpu"
config["device"] = device
config["training::steps_per_epoch"] = 123
module = AEModule(config)

# load checkpoint - adjust the epoch to the one you want to evaluate
checkpoint = torch.load(
    model_path.joinpath("checkpoint_000100/state.pt"), map_location=device
)
module.model.load_state_dict(checkpoint["model"])

# %%
permutation_spec = zoo_cnn_large_permutation_spec()


def load_dataset_from_config(config, epoch_list, permutation_spec, map_to_canonical):
    # load dataset config
    config_ds = json.load(
        open(
            str(config["dataset::dump"]).replace(
                "dataset_beton", "dataset_info_train.json"
            ),
            "r",
        )
    )
    config_ds["tokensize"] = config["ae:i_dim"]
    # get trainset
    trainset = load_single_dataset(
        config_ds=config_ds,
        epoch_list=epoch_list,
        split="train",
        permutation_spec=permutation_spec,
        map_to_canonical=map_to_canonical,
    )
    valset = load_single_dataset(
        config_ds=config_ds,
        epoch_list=epoch_list,
        split="val",
        permutation_spec=permutation_spec,
        map_to_canonical=map_to_canonical,
    )
    testset = load_single_dataset(
        config_ds=config_ds,
        epoch_list=epoch_list,
        split="test",
        permutation_spec=permutation_spec,
        map_to_canonical=map_to_canonical,
    )
    return trainset, valset, testset


def load_single_dataset(
    config_ds, epoch_list, split, permutation_spec, map_to_canonical
):
    # get missing values
    # root = config_ds['zoo_path']
    root = Path(
        ds_info["zoo_path"].replace("[PosixPath(", "").replace(")]", "")[1:-1]
    )  # weird hack but hey
    standardize = config_ds["standardize"]
    ds_split = config_ds["ds_split"]
    max_samples = config_ds["max_samples"]
    weight_threshold = config_ds["weight_threshold"]
    property_keys = {
        "result_keys": [
            "test_acc",
            "training_iteration",
            "ggap",
        ],
        "config_keys": [],
    }
    ignore_bn = False
    tokensize = config_ds["tokensize"]

    # load dataset
    dataset = DatasetTokens(
        root=root,
        epoch_lst=epoch_list,
        permutation_spec=permutation_spec,
        map_to_canonical=map_to_canonical,
        standardize=standardize,
        train_val_test=split,  # determines which dataset split to use
        ds_split=ds_split,  #
        max_samples=max_samples,
        weight_threshold=weight_threshold,
        # filter_function=filter_fn,  # gets sample path as argument and returns True if model needs to be filtered out
        property_keys=property_keys,
        num_threads=12,
        shuffle_path=True,
        verbosity=3,
        getitem="tokens+props",
        ignore_bn=ignore_bn,
        tokensize=tokensize,
    )
    return dataset


def update_json_with_results(json_filename, new_results):
    # curtosy to chatgpt
    # Check if the JSON file already exists
    if os.path.exists(json_filename):
        with open(json_filename, "r") as json_file:
            training_results = json.load(json_file)
    else:
        training_results = []
    # Append the new results to the existing list of training_results
    training_results.append(new_results)
    # Write the updated list back to the JSON file
    with open(json_filename, "w") as json_file:
        json.dump(training_results, json_file, indent=4)


# %%
logging.info("load dataset")
ds_info = json.load(
    open(
        str(config["dataset::dump"]).replace(
            "dataset_beton", "dataset_info_train.json"
        ),
        "r",
    )
)
epoch_list = [1, 3, 5, 10, 15, 20, 25]

# %% if you want to evalaute different epochs, adjust this here
#
# ds_train, ds_val, ds_test = load_dataset_from_config(
#     config,
#     epoch_list=epoch_list,
#     permutation_spec=permutation_spec,
#     map_to_canonical=True,
# )

# use-pre-compiled dataset
ds_path = Path("../data/dataset_cifar100_token_288_ep60_std/")
ds_train = torch.load(ds_path.joinpath("dataset_train.pt"))
ds_test = torch.load(ds_path.joinpath("dataset_test.pt"))
ds_val = torch.load(ds_path.joinpath("dataset_val.pt"))

%%
## Baseline 1: lq
logging.info("compute LQ baseline")
dstk = DownstreamTaskLearner()
lq = LayerQuintiles()

# compute dst perf
performance_lq = dstk.eval_dstasks(
    model=lq,
    trainset=ds_train,
    testset=ds_test,
    valset=ds_val,
    batch_size=config["trainset::batchsize"],
)

performance_lq["method"] = "lq"
update_json_with_results(results_json_path, performance_lq)


%%
# AE
logging.info("compute AE performance")

dstk = DownstreamTaskLearner()

# compute dst perf
performance_ae = dstk.eval_dstasks(
    model=module,
    trainset=ds_train,
    testset=ds_test,
    valset=ds_val,
    batch_size=2,
)

performance_ae["method"] = model_tag

update_json_with_results(results_json_path, performance_ae)