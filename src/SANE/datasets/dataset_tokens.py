import torch
from torch.utils.data import Dataset

from pathlib import Path
import random
import copy

import itertools
from math import factorial

from SANE.datasets.dataset_epochs import ModelDatasetBaseEpochs
from SANE.git_re_basin.git_re_basin import (
    PermutationSpec,
    zoo_cnn_permutation_spec,
    weight_matching,
    apply_permutation,
)

from SANE.models.def_net import NNmodule

from .dataset_auxiliaries import (
    tokenize_checkpoint,
)


import logging

from typing import List, Union, Optional

import ray
from .progress_bar import ProgressBar

import tqdm

from rebasin import PermutationCoordinateDescent


#####################################################################
# Define Dataset class
#####################################################################
class DatasetTokens(ModelDatasetBaseEpochs):
    """
    This class inherits from the base ModelDatasetBaseEpochs class.
    It extends it by permutations of the dataset in the init function.
    """

    # init
    def __init__(
        self,
        root,
        epoch_lst=10,
        mode="vector",  # "vector", "checkpoint"
        permutation_spec: Optional[PermutationSpec] = zoo_cnn_permutation_spec,
        map_to_canonical: bool = False,
        standardize: bool = True,  # wether or not to standardize the data
        tokensize: int = 0,
        train_val_test="train",
        ds_split=[0.7, 0.3],
        weight_threshold: float = float("inf"),
        max_samples: int = 0,  # limit the number of models to integer number (full model trajectory, all epochs)
        filter_function=None,  # gets sample path as argument and returns True if model needs to be filtered out
        property_keys=None,
        shuffle_path: bool = True,
        num_threads=4,
        verbosity=0,
        precision="32",
        supersample: int = 1,  # if supersample > 1, the dataset will be supersampled by this factor
        getitem: str = "tokens",  # "tokens", "tokens+props"
        ignore_bn: bool = False,
        reference_model_path: Union[Path, str] = None,
    ):
        # call init of base class
        super().__init__(
            root=root,
            epoch_lst=epoch_lst,
            mode="checkpoint",
            train_val_test=train_val_test,
            ds_split=ds_split,
            weight_threshold=weight_threshold,
            max_samples=max_samples,
            filter_function=filter_function,
            property_keys=property_keys,
            num_threads=num_threads,
            verbosity=verbosity,
            shuffle_path=shuffle_path,
        )
        self.mode = mode

        self.permutation_spec = permutation_spec
        self.standardize = standardize
        self.tokensize = tokensize
        self.precision = precision
        self.supersample = supersample
        assert self.supersample >= 1

        self.num_threads = num_threads

        self.map_to_canonical = map_to_canonical

        self.ignore_bn = ignore_bn

        self.getitem = getitem

        ### prepare canonical form ###########################################################################################
        if self.map_to_canonical:
            if self.permutation_spec:
                logging.info("prepare canonical form")
                self.map_models_to_canonical()
            else:
                logging.info("prepare canonical form using rebasin")
                self.rebasin()

        ### init len ###########################################################################################
        logging.info("init dataset length")
        self.init_len()

        ### load reference model ###########################################################################################
        if reference_model_path:
            logging.info(f"load reference model from {reference_model_path}")
            self.reference_checkpoint = torch.load(reference_model_path)

        # ### standardize ###################################################################################################################################################################
        if self.standardize == True:
            logging.info("standardize data")
            self.standardize_data_checkpoints()
        elif self.standardize == "minmax":
            logging.info("min max normalization of data")
            self.normalize_data_checkpoints()
        elif self.standardize == "l2_ind":
            self.normalize_checkpoints_separately()

        ### tokensize ###################################################################################################################################################################
        logging.info("tokenize data")
        if self.mode == "vector":
            self.tokenize_data()

        ### set transforms ########################
        logging.info("set transfoorms")
        self.transforms = None

        # set precision
        if self.precision != "32":
            logging.info("set precision")
            if not self.standardize:
                logging.warning(
                    "using lower precision for non-standardized data may cause loss of information"
                )
            self.set_precision(self.precision)

    def tokenize_data(self):
        """
        cast samples as list of tokens to tensors to speed up processing
        """
        # iterate over all samlpes
        for idx in tqdm.tqdm(range(len(self.data)), desc="tokenize model checkpoints"):
            for jdx in range(len(self.data[idx])):
                self.data[idx][jdx], mask, pos = tokenize_checkpoint(
                    checkpoint=self.data[idx][jdx],
                    tokensize=self.tokensize,
                    return_mask=True,
                    ignore_bn=self.ignore_bn,
                )
        # cast to bool to save space
        self.mask = mask.to(torch.bool)
        self.positions = pos.to(torch.int)

    def set_precision(self, precision: str = "32"):
        """ """
        self.precision = precision
        if self.precision == "16":
            dtype = torch.float16
        elif self.precision == "b16":
            dtype = torch.bfloat16
        elif self.precision == "32":
            dtype = torch.float32
        elif self.precision == "64":
            dtype = torch.float64
        else:
            raise NotImplementedError(
                f"precision {self.precision} is not implemented. use 32 or 64"
            )
        # apply precision to weights / tokens
        self.data = [
            [self.data[idx][jdx].to(dtype) for jdx in range(len(self.data[idx]))]
            for idx in range(len(self.data))
        ]

    ## get_weights ####################################################################################################################################################################
    def __get_weights__(
        self,
    ):
        """
        Returns:
            torch.Tensor with full dataset as sequence of components [n_samples,n_tokens_per_sample,token_dim]
        """
        if not self.mode == "vector":
            raise NotImplementedError(
                "mode other than vector is not implemented for DatasetTokens"
            )
        data_out = [
            self.data[idx][jdx]
            for idx in range(len(self.data))
            for jdx in range(len(self.data[idx]))
        ]
        mask_out = [
            self.mask
            for idx in range(len(self.data))
            for jdx in range(len(self.data[idx]))
        ]
        data_out = torch.stack(data_out)
        mask_out = torch.stack(mask_out)
        logging.debug(f"shape of weight tensor: {data_out.shape}")
        return data_out, mask_out

    ## getitem ####################################################################################################################################################################
    def __getitem__(
        self,
        index,
    ):
        """
        Args:
            index (int): Index of the sample to be retrieved
        Returns:
            ddx: OrdereredDict model checkoint or torch.Tensor of neuron tokens with shape [n_tokens_per_sample/windowsize,token_dim]
            mask: optional torch.Tensor of the same shape as ddx_idx indicating the nonzero elements
            pos: optional torch.Tensor with token sequence positions [layer,token_in_layer] of sample
            props: optional torch.Tensor with properties of sample
        """
        # remove supersample (over index, if exists)
        index = index % self._len
        # get model and epoch index
        mdx, edx = self._index[index]
        # get raw data, assume view 1 and view 2 are the same
        ddx = self.data[mdx][edx]
        # get properties
        props = []
        for key in self.property_keys["result_keys"]:
            props.append(self.properties[key][mdx][edx])
        props = torch.tensor(props)
        # check mode
        if self.mode == "vector":
            # data is already tokenized
            # get postion
            pos = self.positions
            # get mask
            mask = self.mask
            if self.getitem == "tokens":
                # return tokens, masks and positions (with or without transforms)
                if self.transforms:
                    ddx, mask, pos = self.transforms(ddx, mask, pos)
                return ddx, mask, pos
            elif self.getitem == "tokens+props":
                # return tokens, masks, positions and properties (with or without transforms)
                if self.transforms:
                    ddx, mask, pos = self.transforms(ddx, mask, pos)
                return ddx, mask, pos, props
            else:
                raise NotImplementedError(f"getitem {self.getitem} is not implemented")
        else:
            # operate on checkpoint
            if self.getitem == "tokens":
                # just checkpoints, with or without transforms
                if not self.transforms:
                    return ddx
                else:
                    return self.transforms(ddx)
            elif self.getitem == "tokens+props":
                # checkpoints and properties, with or without transforms
                if not self.transforms:
                    return ddx, props
                else:
                    return self.transforms(ddx, props)
            else:
                raise NotImplementedError(f"getitem {self.getitem} is not implemented")

    ### len ##################################################################################################################################################################
    def init_len(self):
        index = []
        for idx, ddx in enumerate(self.data):
            idx_tmp = [(idx, jdx) for jdx in range(len(ddx))]
            index.extend(idx_tmp)
        self._len = len(index)
        self._index = index

    def __len__(self):
        if self.supersample:
            # supersampling extends the len of the dataset by the sumpersample factor
            # the dataset will be iterated over several times.
            # this only makes sense if transformations are used
            # that way, it extends one epoch and reduces dataloading / synchronization epoch
            # motivation: if samples are sliced to small subsets, using the same sample multiple times may reduce overhead
            return self._len * self.supersample
        return self._len

    def standardize_data_checkpoints(self):
        """
        standardize data to zero-mean / unit std. per layer
        store per-layer mean / std
        """
        logging.info("Get layer mapping")
        # step 1: get token-layer index relation
        layers = {}
        # iterate over layers
        chkpt = self.reference_checkpoint
        for key in tqdm.tqdm(chkpt.keys(), desc="standardize model weights per layer"):
            # if "weight" in key:
            #     #### get weights ####
            #     if self.ignore_bn and ("bn" in key or "downsample.1" in key):
            #         continue
            # filter out bn layers if ignore_bn is set
            if "bn" in key or "downsample.1" in key:
                if self.ignore_bn:
                    continue
            # get weights of all layers, including bn running mean / var
            if "weight" in key or "running_mean" in key or "running_var" in key:
                # first iteration over data: get means / std
                means = []
                stds = []
                for idx in range(len(self.data)):
                    for jdx in range(len(self.data[idx])):
                        w = self.data[idx][jdx][key]
                        # flatten to out_channels x n
                        w = w.view(w.shape[0], -1)
                        # cat biases to channels if they exist in self.data[idx][jdx]
                        if key.replace("weight", "bias") in self.data[idx][jdx]:
                            b = self.data[idx][jdx][key.replace("weight", "bias")]
                            w = torch.cat([w, b.unsqueeze(dim=1)], dim=1)
                        # compute masked mean / std
                        tmp_mean = w.mean()
                        tmp_std = w.std()
                        # append values to lists
                        means.append(tmp_mean.item())
                        stds.append(tmp_std.item())

                # aggregate means / stds
                # compute mean / std assuming all samples have the same numels
                mu = torch.mean(torch.tensor(means))
                sigma = torch.sqrt(torch.mean(torch.tensor(stds) ** 2))

                # store in layer
                layers[key] = {}
                layers[key]["mean"] = mu.item()
                layers[key]["std"] = sigma.item()

                # secon iteration over data: standardize
                for idx in range(len(self.data)):
                    for jdx in range(len(self.data[idx])):
                        # normalize weights
                        self.data[idx][jdx][key] = (
                            self.data[idx][jdx][key] - mu
                        ) / sigma
                        # normalize biases if they exist
                        if key.replace("weight", "bias") in self.data[idx][jdx]:
                            self.data[idx][jdx][key.replace("weight", "bias")] = (
                                self.data[idx][jdx][key.replace("weight", "bias")] - mu
                            ) / sigma

        self.layers = layers

    def normalize_data_checkpoints(self):
        """
        min-max normalization of data to zero-mean / unit std. per layer
        store per-layer mean / std
        """
        logging.info("Get layer mapping")
        # step 1: get token-layer index relation
        layers = {}
        # iterate over layers
        chkpt = self.reference_checkpoint
        for key in tqdm.tqdm(chkpt.keys(), desc="normalize model weights per layer"):
            if "weight" in key:
                # get weights
                if self.ignore_bn and ("bn" in key or "downsample.1" in key):
                    continue
                # TODO: add running mean / var?
                # first iteration over data: get means / std
                mins = []
                maxs = []
                for idx in range(len(self.data)):
                    for jdx in range(len(self.data[idx])):
                        w = self.data[idx][jdx][key]
                        # flatten to out_channels x n
                        w = w.view(w.shape[0], -1)
                        # cat biases to channels if they exist in self.data[idx][jdx]
                        if key.replace("weight", "bias") in self.data[idx][jdx]:
                            b = self.data[idx][jdx][key.replace("weight", "bias")]
                            w = torch.cat([w, b.unsqueeze(dim=1)], dim=1)
                        # compute masked mean / std
                        tmp_min = w.flatten().min()
                        tmp_max = w.flatten().max()
                        # append values to lists
                        mins.append(tmp_min.item())
                        maxs.append(tmp_max.item())
                # aggregate means / stds
                # compute mean / std assuming all samples have the same numels
                min_glob = torch.min(torch.tensor(mins))
                max_glob = torch.max(torch.tensor(maxs))

                # store in layer
                layers[key] = {}
                layers[key]["min"] = min_glob.item()
                layers[key]["max"] = max_glob.item()

                # secon iteration over data: min-max normalization
                for idx in range(len(self.data)):
                    for jdx in range(len(self.data[idx])):
                        # normalize weights
                        self.data[idx][jdx][key] = (
                            self.data[idx][jdx][key] - min_glob
                        ) / (max_glob - min_glob) * 2 - 1
                        # normalize biases if they exist
                        if key.replace("weight", "bias") in self.data[idx][jdx]:
                            self.data[idx][jdx][key.replace("weight", "bias")] = (
                                self.data[idx][jdx][key.replace("weight", "bias")]
                                - min_glob
                            ) / (max_glob - min_glob) * 2 - 1

        self.layers = layers

    def normalize_checkpoints_separately(self):
        """
        l2-normalization of all checkoints and layers, individually (note: no shared normalization coeff)
        we currently don't keep the normalization coefficients, as we don't plan to reconstruct ever.
        This may need fixing in the future.
        """
        logging.info("apply l2 normalization")
        # iterate over data points
        for idx in tqdm.tqdm(
            range(len(self.data)), desc="normalize model weights per model"
        ):
            for jdx in range(len(self.data[idx])):
                # iterate over layers:
                for key in self.data[idx][jdx].keys():
                    if "weight" in key:
                        #### get weights ####
                        if self.ignore_bn and ("bn" in key or "downsample.1" in key):
                            continue
                        w = self.data[idx][jdx][key]
                        # flatten to out_channels x n
                        w = w.view(w.shape[0], -1)
                        # cat biases to channels if they exist in self.data[idx][jdx]
                        if key.replace("weight", "bias") in self.data[idx][jdx]:
                            b = self.data[idx][jdx][key.replace("weight", "bias")]
                            w = torch.cat([w, b.unsqueeze(dim=1)], dim=1)
                        # compute l2 norm of flattened weights
                        l2w = torch.norm(w.flatten(), p=2, dim=0)
                        # normalize weights with l2w
                        self.data[idx][jdx][key] = self.data[idx][jdx][key] / l2w
                        # normalize biases with l2w
                        if key.replace("weight", "bias") in self.data[idx][jdx]:
                            self.data[idx][jdx][key.replace("weight", "bias")] = (
                                self.data[idx][jdx][key.replace("weight", "bias")] / l2w
                            )

    ### map data to canoncial #############################################################################################
    def map_models_to_canonical(self):
        """
        define reference model
        iterate over all models
        get permutation w.r.t last epoch (best convergence)
        apply same permutation on all epochs (on raw data)
        """
        # use first model / last epoch as reference model (might be sub-optimal)
        reference_model = self.reference_checkpoint

        ## init multiprocessing environment ############
        ray.init(num_cpus=self.num_threads)

        ### gather data #############################################################################################
        print(f"preparing computing canon form...")
        pb = ProgressBar(total=len(self.data))
        pb_actor = pb.actor

        for idx in range(len(self.data)):
            # align models using git-re-basin
            perm_spec = self.permutation_spec
            # get second
            model_curr = self.data[idx]
            model_curr = compute_single_canon_form.remote(
                reference_model=reference_model,
                data_curr=model_curr,
                perm_spec=perm_spec,
                pba=pb_actor,
            )
            self.data[idx] = model_curr

        # update progress bar
        pb.print_until_done()

        self.data = [ray.get(self.data[idx]) for idx in range(len(self.data))]
        ray.shutdown()

    ### map data to canoncial #############################################################################################
    def rebasin(self):
        """
        define reference model
        iterate over all models
        get permutation w.r.t last epoch (best convergence)
        apply same permutation on all epochs (on raw data)
        """
        # use first model / last epoch as reference model (might be sub-optimal)
        reference_check = self.reference_checkpoint

        # create reference data
        ds_downstream = torch.load(self.reference_config["dataset::dump"])
        trainset = ds_downstream["trainset"]
        s1 = trainset.__getitem__(0)
        s2 = trainset.__getitem__(1)
        data_sample = torch.stack(s1[0], s2[0])

        ## init multiprocessing environment ############
        ray.init(num_cpus=self.num_threads)

        ### gather data #############################################################################################
        print(f"preparing computing canon form...")
        pb = ProgressBar(total=len(self.data))
        pb_actor = pb.actor

        for idx in range(len(self.data)):
            # align models using git-re-basin
            model_curr = self.data[idx]
            model_curr = compute_single_rebasin.remote(
                reference_check=reference_check,
                data_curr=model_curr,
                config=self.reference_config,
                data_sample=data_sample,
                pba=pb_actor,
            )
            self.data[idx] = model_curr

        # update progress bar
        pb.print_until_done()

        self.data = [ray.get(self.data[idx]) for idx in range(len(self.data))]
        ray.shutdown()


### helper parallel function #############################################################################################
@ray.remote(num_returns=1)
def compute_single_canon_form(reference_model, data_curr, perm_spec, pba):
    # get second
    model_curr = data_curr[-1]
    # find permutation to match params_b to params_a
    logging.debug(
        f"compute canonical form: params a {type(reference_model)} params b {type(model_curr)}"
    )
    match_permutation = weight_matching(
        ps=perm_spec, params_a=reference_model, params_b=model_curr
    )
    # apply permutation on all epochs
    for jdx in range(len(data_curr)):
        model_curr = data_curr[jdx]
        model_curr_perm = apply_permutation(
            ps=perm_spec, perm=match_permutation, params=model_curr
        )
        # put back in data
        data_curr[jdx] = model_curr_perm

    # update counter
    pba.update.remote(1)
    # return list
    return data_curr


### helper parallel function #############################################################################################
@ray.remote(num_returns=1)
def compute_single_rebasin(reference_check, data_curr, config, data_sample, pba):
    # instantiate two models
    model_a = NNmodule(config)
    model_b = NNmodule(config)
    # load checkpoints
    model_a.model.load_state_dict(reference_check)
    # find permutation to match params_b to params_a
    logging.debug(
        f"compute canonical form: params a {type(reference_check)} params b {type(model_curr)}"
    )
    # todo
    input_data = data_sample

    # apply permutation on all epochs
    for jdx in range(len(data_curr)):
        check_curr = data_curr[jdx]
        # Rebasin
        model_b.model.load_state_dict(check_curr)
        pcd = PermutationCoordinateDescent(
            model_a, model_b, input_data
        )  # weight-matching
        pcd.rebasin()  # Rebasin model_b towards model_a. Automatically updates model_b

        # put back in data
        data_curr[jdx] = model_b.model.state_dict()

    # update counter
    pba.update.remote(1)
    # return list
    return data_curr
