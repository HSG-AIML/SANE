from ffcv.writer import DatasetWriter
from ffcv.fields import NDArrayField, FloatField, TorchTensorField, BytesField

from typing import Union, List, Tuple, Dict, Any

from pathlib import Path

from SANE.git_re_basin.git_re_basin import PermutationSpec

from SANE.datasets.dataset_tokens import DatasetTokens
from SANE.datasets.augmentations import (
    WindowCutter,
    TokenizerAugmentation,
    CheckpointAugmentationPipeline,
)

import logging

import json

import torch


def prepare_ffcv_dataset(
    dataset_target_path: Union[str, Path],
    zoo_path: Union[list, str, Path],
    epoch_list: list,
    permutation_spec: PermutationSpec,
    map_to_canonical: bool = True,
    standardize: bool = True,
    ds_split: list = [0.7, 0.15, 0.15],
    splits: list = ["train", "val", "test"],
    max_samples: int = 1000,
    weight_threshold: int = 15,
    property_keys: dict = {
        "result_keys": [
            "test_acc",
            "training_iteration",
            "ggap",
        ],
        "config_keys": [],
    },
    filter_fn: Any = None,
    num_threads: int = 12,
    shuffle_path: bool = True,
    windowsize: int = 160,
    supersample: Union[str, int] = "auto",
    precision: int = 16,
    ignore_bn: bool = True,
    tokensize: int = 576,
    permutation_number_train: int = 0,
    permutations_per_sample_train: int = 0,
    permutation_number_test: int = 0,
    permutations_per_sample_test: int = 0,
    page_size: int = 4 * 1 << 21,
    drop_pt_dataset: bool = False,
):
    """
    Prepares an ffcv dataset from token_dataset.
    Args:
        dataset_target_path: Path to the target dataset.
        zoo_path: Path to the zoo.
        epoch_list: List of epochs to use.
        permutation_spec: PermutationSpec to use.
        map_to_canonical: Whether to map models to canonical from using git-rebasin.
        standardize: Whether to standardize the weights (per layer).
        ds_split: Dataset split, in "train" "val" "test".
        max_samples: Maximum number of samples, split by model path to prevent leakage, distributed over splits.
        weight_threshold: Weight threshold in 1-norm.
        property_keys: Property keys (load properties).
        filter_fn: function to filter out models with
        num_threads: Number of threads.
        shuffle_path: Whether to shuffle the path.
        supersample: Supersample.
        ignore_bn: weather to load batchnorm paramters
        tokensize: set dimension of tokens. set to 0 to discover size.
        permutation_number_train: Number of permutations to use for training.
        permutations_per_sample_train: Number of permutations per sample to use for training.
        permutation_number_test: Number of permutations to use for testing.
        permutations_per_sample_test: Number of permutations per sample to use for testing.
        page_size: ffcv Page size, see below.
        drop_pt_dataset: flag wheater to write out the dataset.pt torch.utils.data.Dataset type dataset as pickle as well.
    Returns:
        None
    """

    # load conventional datasets

    for split_key in splits:
        logging.info(f"load {split_key} dataset")
        permutation_number = (
            permutation_number_train
            if split_key == "train"
            else permutation_number_test
        )
        permutations_per_sample = (
            permutations_per_sample_train
            if split_key == "train"
            else permutations_per_sample_test
        )
        preprocess_single_split(
            dataset_target_path=dataset_target_path,
            zoo_path=zoo_path,
            epoch_list=epoch_list,
            permutation_spec=permutation_spec,
            map_to_canonical=map_to_canonical,
            permutation_number=permutation_number,
            permutations_per_sample=permutations_per_sample,
            standardize=standardize,
            ds_split=ds_split,
            max_samples=max_samples,
            weight_threshold=weight_threshold,
            property_keys=property_keys,
            filter_fn=filter_fn,
            num_threads=num_threads,
            shuffle_path=shuffle_path,
            windowsize=windowsize,
            supersample=supersample,
            precision=precision,
            split=split_key,
            ignore_bn=ignore_bn,
            tokensize=tokensize,
            page_size=page_size,
            drop_pt_dataset=drop_pt_dataset,
        )


def preprocess_single_split(
    dataset_target_path: Union[str, Path],
    zoo_path: Union[list, str, Path],
    epoch_list: list,
    permutation_spec: PermutationSpec,
    map_to_canonical: bool = True,
    permutation_number: int = 0,
    permutations_per_sample: int = 0,
    standardize: bool = True,
    ds_split: list = [0.7, 0.15, 0.15],
    max_samples: int = 1000,
    weight_threshold=15,
    property_keys: dict = {
        "result_keys": [
            "test_acc",
            "training_iteration",
            "ggap",
        ],
        "config_keys": [],
    },
    filter_fn: Any = None,
    num_threads: int = 12,
    shuffle_path: bool = True,
    windowsize: int = 160,
    supersample: Union[str, int] = "auto",
    precision: str = "16",
    split: str = "train",
    ignore_bn: bool = True,
    tokensize: int = 576,
    page_size: int = 4 * 1 << 21,
    drop_pt_dataset: bool = False,
):
    """
    Loads a single split of the token dataset and writes to ffcv.
    Args:
        dataset_target_path: Path to the target dataset.
        zoo_path: Path to the zoo.
        epoch_list: List of epochs to use.
        permutation_spec: PermutationSpec to use.
        map_to_canonical: Whether to map models to canonical from using git-rebasin.
        permutation_number: Number of permutations to prepare.
        permutations_per_sample: Number of permutations per sample (each sample is a stack of several permuted versions).
        standardize: Whether to standardize the weights (per layer).
        ds_split: Dataset split, in "train" "val" "test".
        max_samples: Maximum number of samples, split by model path to prevent leakage, distributed over splits.
        weight_threshold: Weight threshold in 1-norm.
        property_keys: Property keys (load properties).
        filter_fn: function to filter out models with
        num_threads: Number of threads.
        shuffle_path: Whether to shuffle the path.
        windowsize: Windowsize.
        supersample: Supersample.
        split: Split to use.
        ignore_bn: weather to load batchnorm paramters
        tokensize: set dimension of tokens. set to 0 to discover size.
        page_size: ffcv paramter size of page to use for mmap used in dataset writer
        drop_pt_dataset: weather or not to aditionally write the torch.utils.Dataset type as .pt file as well.

    Returns:
        None
    """

    # check type of permutation_spec
    if callable(permutation_spec):
        permutation_spec = permutation_spec()

    if isinstance(zoo_path, list):
        root = [Path(pdx).absolute() for pdx in zoo_path]
    else:
        root = Path(zoo_path).absolute()

    print("Load token dataset")
    logging.info("Load token dataset")
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
        precision=precision,
        filter_function=filter_fn,  # gets sample path as argument and returns True if model needs to be filtered out
        property_keys=property_keys,
        num_threads=12,
        shuffle_path=True,
        verbosity=3,
        mode="checkpoint",  # apply permutation on checkpoint
        getitem="tokens+props",
        ignore_bn=ignore_bn,
        tokensize=tokensize,
    )

    # check existance of dataset parth
    Path(dataset_target_path).mkdir(parents=True, exist_ok=True)

    # drop pt file
    if drop_pt_dataset:
        write_path = Path(dataset_target_path).joinpath(f"dataset_{split}.pt")
        try:
            torch.save(dataset, write_path)
        except Exception as e:
            logging.error(f"could not save dataset.pt at {write_path}")
            logging.error(e)

    # set windowcutter transform
    logging.info("set augmentations before ffcv dataset")
    dataset.transforms = CheckpointAugmentationPipeline(
        perm_spec=permutation_spec,
        tokensize=dataset.tokensize,
        ignore_bn=ignore_bn,
        permutation_number=permutation_number,
        windowsize=windowsize,
    )

    # set supersample
    if supersample == "auto":
        # infer number of iterations over each sample as len of token sequence divided by windowsize
        supersample = dataset.data[0][0].shape[0] // windowsize
    logging.info(f"set to supersample: {supersample}")

    # set supersample in the dataset
    dataset.supersample = supersample

    logging.info(f"dataset len: {len(dataset)}")

    # get sample and infer dimensions
    logging.info("get sample and infer dimensions")
    # ddx, mask, pos = dataset.__getitem__(0)
    ddx, mask, pos, props = dataset.__getitem__(0)

    logging.info(f"ddx.shape: {ddx.shape} - dtype: {ddx.dtype}")
    logging.info(f"mask.shape: {mask.shape} - dtype: {mask.dtype}")
    logging.info(f"pos.shape: {pos.shape} - dtype: {pos.dtype}")
    logging.info(f"props.shape: {props.shape} - dtype: {props.dtype}")

    # configure writer
    logging.info("configure ffcv writer")
    # """
    write_path = Path(dataset_target_path).joinpath(f"dataset_beton.{split}")
    writer = DatasetWriter(
        write_path,
        {
            "w": TorchTensorField(
                shape=ddx.shape, dtype=ddx.dtype
            ),  # torch.float32 or 16
            "m": TorchTensorField(shape=mask.shape, dtype=mask.dtype),  # torch.bool
            "p": TorchTensorField(shape=pos.shape, dtype=pos.dtype),  # torch.int16
            "props": TorchTensorField(
                shape=props.shape, dtype=props.dtype
            ),  # torch.float32
        },
        page_size=page_size,
        num_workers=16,
    )
    # write dataset
    logging.info("write ffcv dataset to disk")
    writer.from_indexed_dataset(dataset)

    # get full sample and infer dimensions
    dataset.transforms = CheckpointAugmentationPipeline(
        perm_spec=permutation_spec,
        tokensize=dataset.tokensize,
        ignore_bn=ignore_bn,
        permutation_number=permutation_number,
        windowsize=1000000000,  # set windowsize very large
    )
    ddx, mask, pos, props = dataset.__getitem__(0)

    # drop info
    logging.info("collect info and write to disk")
    info = {
        "zoo_path": str(zoo_path),
        "num_samples": dataset._len,
        "supersample": supersample,
        "properties": dataset.property_keys["result_keys"],
        "epoch_list": epoch_list,
        "map_to_canonical": map_to_canonical,
        "permutation_number": permutation_number,
        "permutations_per_sample": permutations_per_sample,
        "standardize": standardize,
        "ds_split": ds_split,
        "max_samples": max_samples,
        "weight_threshold": weight_threshold,
        "property_keys": property_keys,
        "num_threads": num_threads,
        "shuffle_path": shuffle_path,
        "windowsize": windowsize,
        "split": split,
        "max_positions": pos.max(dim=0).values.tolist(),
    }
    # add info json to the same path
    json_path = Path(dataset_target_path).joinpath(f"dataset_info_{split}.json")
    json.dump(info, json_path.open("w"))
    # """

    # dump normalization data
    try:
        layer_norms = dataset.layers
    except Exception as e:
        print(e)
        print("no norm found")
        layer_norms = {}
    if standardize == True:
        norm_mode = "standardize"
    elif standardize == "minmax":
        norm_mode = "minmax"
    else:
        norm_mode = None
    layer_norms["mode"] = norm_mode
    # add info json to the same path
    json_path = Path(dataset_target_path).joinpath(
        f"dataset_normalization_{split}.json"
    )
    json.dump(layer_norms, json_path.open("w"))
    # """

    # test dataset
    logging.info("test dataset with dataloader")
    from ffcv.loader import Loader, OrderOption

    batch_size = 64
    num_workers = 4
    ordering = OrderOption.QUASI_RANDOM
    # Dataset ordering

    from ffcv.fields.decoders import NDArrayDecoder
    from ffcv.transforms import ToTensor, Convert

    PIPELINES = {
        "w": [NDArrayDecoder(), ToTensor(), Convert(torch.float16)],
        "m": [NDArrayDecoder(), ToTensor()],
        "p": [NDArrayDecoder(), ToTensor()],
        "props": [NDArrayDecoder(), ToTensor()],
    }

    loader = Loader(
        write_path,
        batch_size=batch_size,
        num_workers=num_workers,
        order=ordering,
        drop_last=True,
        pipelines=PIPELINES,
        os_cache=False,
    )
    # save params.json
    path_list = [pdx for pdx in root[0].iterdir() if pdx.is_dir()]
    json_path = path_list[0].joinpath("params.json")
    model_example_config = json.load(json_path.open("r"))
    # write json to target path
    json_path_target = Path(dataset_target_path).joinpath("params.json")
    # dump json to target
    json.dump(model_example_config, json_path_target.open("w"))

    print(f"loader len: {len(loader)}")
    for idx, (ddx, mask, pos, props) in enumerate(loader):
        print(f"ddx.shape: {ddx.shape} - dtype: {ddx.dtype}")
        print(f"mask.shape: {mask.shape} - dtype: {mask.dtype}")
        print(f"pos.shape: {pos.shape} - dtype: {pos.dtype}")
        print(f"props.shape: {props.shape} - dtype: {props.dtype}")
        # if idx == 10:
        break
