from collections import OrderedDict
from SANE.datasets.dataset_auxiliaries import tokens_to_checkpoint, tokenize_checkpoint
from SANE.datasets.def_FastTensorDataLoader import FastTensorDataLoader
from SANE.models.def_NN_experiment import NNmodule

import torch

from typing import Optional, List, Any

import numpy as np

from einops import repeat

from sklearn.neighbors import KernelDensity

import logging

from SANE.sampling.halo import haloify, dehaloify

from SANE.sampling.condition_bn import condition_checkpoints, check_equivalence


def de_normalize_checkpoint(checkpoint, layers, mode="minmax"):
    """
    revert normalization
    """
    # iterate over layer keys instead of checkpoint keys
    # that way, we only consider layers for which we have norm values
    for key in layers.keys():
        if key == "mode":
            continue
        if mode == "standardize":
            # get mean and std
            mu = layers[key]["mean"]
            sigma = layers[key]["std"]
            # de-normalize weights
            checkpoint[key] = checkpoint[key] * sigma + mu
            # de-noramlize bias
            if key.replace("weight", "bias") in checkpoint:
                checkpoint[key.replace("weight", "bias")] = (
                    checkpoint[key.replace("weight", "bias")] * sigma + mu
                )
        elif mode == "minmax":
            # get global min and max values
            min_glob = layers[key]["min"]
            max_glob = layers[key]["max"]
            # reverse of min-max normalization (mapped to range [-1,1])
            # returns weights exactly to original range
            checkpoint[key] = (checkpoint[key] + 1) * (
                max_glob - min_glob
            ) / 2 + min_glob
            # de-normalize bais
            if key.replace("weight", "bias") in checkpoint:
                checkpoint[key.replace("weight", "bias")] = (
                    checkpoint[key.replace("weight", "bias")] + 1
                ) * (max_glob - min_glob) / 2 + min_glob

    return checkpoint
