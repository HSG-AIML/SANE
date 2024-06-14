from pathlib import Path

import torch

from torch.utils.data import Dataset


from SANE.datasets.dataset_auxiliaries import (
    test_checkpoint_for_nan,
    test_checkpoint_with_threshold,
)

import random
import copy
import json
import tqdm


import ray
from .progress_bar import ProgressBar

import logging


class ModelDatasetBaseEpochs(Dataset):
    """
    This dataset class loads checkpoints from path, stores them in the memory
    It considers different task, but does not implement the __getitem__ function or any augmentation
    Each item in the list is the full trajectory over all epochs
    """

    # init
    def __init__(
        self,
        root,
        epoch_lst=10,
        mode="checkpoint",
        train_val_test="train",  # determines whcih dataset split to use
        ds_split=[0.7, 0.3],  #
        max_samples=None,
        weight_threshold=float("inf"),
        filter_function=None,  # gets sample path as argument and returns True if model needs to be filtered out
        property_keys=None,
        num_threads=4,
        shuffle_path=True,
        verbosity=0,
    ):
        self.epoch_lst = epoch_lst
        self.mode = mode
        self.verbosity = verbosity
        self.weight_threshold = weight_threshold
        self.property_keys = copy.deepcopy(property_keys)
        self.train_val_test = train_val_test
        self.ds_split = ds_split

        ### initialize data over epochs #####################
        if not isinstance(epoch_lst, list):
            epoch_lst = [epoch_lst]
        # get iterator for epochs
        eldx_lst = range(len(epoch_lst))

        ### prepare directories and path list ################################################################

        ## check if root is list. if not, make root a list
        if not isinstance(root, list):
            root = [root]

        ## make path an absolute pathlib Path
        for rdx in root:
            if isinstance(rdx, str):
                rdx = Path(rdx)
        self.root = root

        # get list of folders in directory
        self.path_list = []
        for rdx in self.root:
            pth_lst_tmp = [f for f in rdx.iterdir() if f.is_dir()]
            self.path_list.extend(pth_lst_tmp)

        # shuffle self.path_list
        if shuffle_path:
            random.seed(42)
            random.shuffle(self.path_list)

        ### Split Train and Test set ###########################################################################
        if max_samples is not None:
            self.path_list = self.path_list[:max_samples]

        ### get reference model ###########################################################################
        # iterate over path list
        for pdx in self.path_list:
            for edx in range(len(epoch_lst)):
                ep = epoch_lst[-edx - 1]
                # try to load model at last epoch
                # first successsful load becomes reference checkpoint
                ref_check, ref_lab, ref_path, ref_ep = load_checkpoint(
                    path=pdx,  # path to model
                    edx=ep,  # checkpoint_epoch/number
                    weight_threshold=self.weight_threshold,
                    filter_function=filter_function,
                )
                if ref_check is not None:
                    break
            if ref_check is not None:
                break

        self.reference_checkpoint = copy.deepcopy(ref_check)
        self.reference_label = copy.deepcopy(ref_lab)
        self.reference_path = copy.deepcopy(ref_path)
        self.reference_epoch = copy.deepcopy(ref_ep)

        config_path = ref_path.joinpath("params.json")
        ref_config = json.load(config_path.open("r"))
        self.reference_config = ref_config

        assert self.reference_checkpoint is not None, "no reference checkpoint found"
        logging.info(f"reference checkpoint found at {self.reference_path}")

        ### Split Train and Test set ###########################################################################
        assert (
            abs(sum(self.ds_split) - 1.0) < 1e-8
        ), f"dataset splits {self.ds_split} sum up to {sum(self.ds_split)} but should equal to 1"
        # two splits
        if len(self.ds_split) == 2:
            if self.train_val_test == "train":
                idx1 = int(self.ds_split[0] * len(self.path_list))
                self.path_list = self.path_list[:idx1]
            elif self.train_val_test == "test":
                idx1 = int(self.ds_split[0] * len(self.path_list))
                self.path_list = self.path_list[idx1:]
            else:
                logging.error(
                    "validation split requested, but only two splits provided."
                )
                raise NotImplementedError(
                    "validation split requested, but only two splits provided."
                )
        # three splits
        elif len(self.ds_split) == 3:
            if self.train_val_test == "train":
                idx1 = int(self.ds_split[0] * len(self.path_list))
                self.path_list = self.path_list[:idx1]
            elif self.train_val_test == "val":
                idx1 = int(self.ds_split[0] * len(self.path_list))
                idx2 = idx1 + int(self.ds_split[1] * len(self.path_list))
                self.path_list = self.path_list[idx1:idx2]
            elif self.train_val_test == "test":
                idx1 = int(self.ds_split[0] * len(self.path_list))
                idx2 = idx1 + int(self.ds_split[1] * len(self.path_list))
                self.path_list = self.path_list[idx2:]
        else:
            logging.warning(f"dataset splits are unintelligble. Load 100% of dataset")
            pass

        ### prepare data lists ###############
        data = []
        labels = []
        paths = []
        epochs = []

        ## init multiprocessing environment ############
        ray.init(num_cpus=num_threads)

        ### gather data #############################################################################################
        logging.info(f"loading checkpoints from {self.root}")
        pb = ProgressBar(total=len(self.path_list * len(eldx_lst)))
        pb_actor = pb.actor
        for idx, path in enumerate(self.path_list):
            ## iterate over all epochs in list
            model_data = []
            model_labels = []
            model_paths = []
            model_epochs = []

            for edx in epoch_lst:
                # # call function in parallel
                (
                    ddx,
                    ldx,
                    path_dx,
                    epoch_dx,
                ) = load_checkpoints_remote.remote(
                    path=path,
                    edx=edx,
                    weight_threshold=self.weight_threshold,
                    filter_function=filter_function,
                    verbosity=self.verbosity,
                    pba=pb_actor,
                )
                # append returns to lists per model
                model_data.append(ddx)
                model_labels.append(ldx)
                model_paths.append(path_dx)
                model_epochs.append(epoch_dx)

            # append lists per model to overall lists
            data.append(model_data)
            labels.append(model_labels)
            paths.append(model_paths)
            epochs.append(model_epochs)

        pb.print_until_done()

        # collect actual data
        data = [ray.get(ddx) for ddx in data]
        labels = [ray.get(ldx) for ldx in labels]
        paths = [ray.get(pdx) for pdx in paths]
        epochs = [ray.get(edx) for edx in epochs]
        # data = ray.get(data)
        # labels = ray.get(labels)
        # paths = ray.get(paths)
        # epochs = ray.get(epochs)

        ray.shutdown()

        # remove None values
        data = [[ddx for ddx in data[idx] if ddx] for idx in range(len(data))]
        labels = [[ddx for ddx in labels[idx] if ddx] for idx in range(len(labels))]
        epochs = [[ddx for ddx in epochs[idx] if ddx] for idx in range(len(epochs))]
        paths = [[ddx for ddx in paths[idx] if ddx] for idx in range(len(paths))]
        # remove empty values
        data = [ddx for ddx in data if len(ddx) > 0]
        labels = [ddx for ddx in labels if len(ddx) > 0]
        epochs = [ddx for ddx in epochs if len(ddx) > 0]
        paths = [ddx for ddx in paths if len(ddx) > 0]

        self.data = copy.deepcopy(data)
        self.labels = copy.deepcopy(labels)
        self.paths = copy.deepcopy(paths)
        self.epochs = copy.deepcopy(epochs)

        logging.info(
            f"Data loaded. found {len(self.data)} usable samples out of potential {len(self.path_list * len(eldx_lst))} samples."
        )

        if self.property_keys is not None:
            logging.info(f"Load properties for samples from paths.")

            # get propertys from path
            result_keys = self.property_keys.get("result_keys", [])
            config_keys = self.property_keys.get("config_keys", [])
            # figure out offset
            try:
                self.read_properties(
                    results_key_list=result_keys,
                    config_key_list=config_keys,
                    idx_offset=0,
                )
            except AssertionError as e:
                logging.error(f"Exception occurred: {e}", exc_info=True)
                self.read_properties(
                    results_key_list=result_keys,
                    config_key_list=config_keys,
                    idx_offset=1,
                )
            logging.info(f"Properties loaded.")
        else:
            self.properties = None

    ## getitem ####################################################################################################################################################################
    def __getitem__(self, index):
        # not implemented in base class
        raise NotImplementedError(
            "the __getitem__ function is not implemented in the base class. "
        )
        pass

    ## len ####################################################################################################################################################################
    def __len__(self):
        return len(self.data)

    ### get_weights ##################################################################################################################################################################
    """
    def __get_weights__(self):
        if self.mode == "checkpoint":
            self.vectorize_data()
        else:
            self.weights = torch.stack(self.data, dim=0)
        return self.weights
    """

    ### get_weights_per_channel ##################################################################################################################################################################
    """
    def __get_weights_per_channel__(self):
        if self.mode == "checkpoint":
            self.vectorize_per_channel()
        elif self.mode == "vector" or self.mode == "vectorized":
            raise NotImplementedError(
                "extracting weights per layer from vector form is not (yet) implemented in the base class. "
            )
        return self.weights
    """

    ## read properties from path ##############################################################################################################################################
    def read_properties(self, results_key_list, config_key_list, idx_offset=1):
        # copy results_key_list to prevent kickback of delete to upstream function
        results_key_list = [key for key in results_key_list]
        # init dict
        properties = {}
        for key in results_key_list:
            properties[key] = []
        for key in config_key_list:
            properties[key] = []
        # remove ggap from results_key_list -> cannot be read, has to be computed.
        read_ggap = False
        if "ggap" in results_key_list:
            results_key_list.remove("ggap")
            read_ggap = True

        properties_template = copy.deepcopy(properties)

        logging.info(f"### load data for {properties.keys()}")

        # iterate over models
        for iidx, ppdx in tqdm.tqdm(enumerate(self.paths), desc="load properties"):
            # instanciate model lists
            properties_temp = copy.deepcopy(properties_template)
            # iterate over epochs in model
            for jjdx, eedx in enumerate(self.epochs[iidx]):
                pathdx = ppdx[jjdx]
                res_tmp = read_properties_from_path(pathdx, eedx, idx_offset=idx_offset)
                for key in results_key_list:
                    properties_temp[key].append(res_tmp[key])
                for key in config_key_list:
                    properties_temp[key].append(res_tmp["config"][key])
                # compute ggap
                if read_ggap:
                    gap = res_tmp["train_acc"] - res_tmp["test_acc"]
                    properties_temp["ggap"].append(gap)
                # assert epoch == training_iteration -> match correct data
                if iidx == 0:
                    train_it = int(res_tmp["training_iteration"])
                    assert (
                        int(eedx) == train_it
                    ), f"training iteration {train_it} and epoch {eedx} don't match."

            # transfer model lists to properties dict
            for key in results_key_list:
                properties[key].append(properties_temp[key])
            for key in config_key_list:
                properties[key].append(properties_temp[key])
            # compute ggap
            if read_ggap:
                properties["ggap"].append(properties_temp["ggap"])

        self.properties = properties

    ## vectorize data ##################################################################################
    """
    def vectorize_per_channel(self):
        # save base checkpoint (as later reference)
        self.checkpoint_base = self.data[0]
        # iterate over length of dataset
        self.weights = []
        for idx in tqdm.tqdm(range(self.__len__())):
            checkpoint = copy.deepcopy(self.data[idx])
            ddx, channel_labels = extract_weights_per_layer_from_checkpoint(checkpoint)
            self.weights.append(ddx)
        self.channel_labels = channel_labels
    """

    ## get params data ##################################################################################
    """
    def vectorize_data(self):
        # save base checkpoint (as later reference)
        self.checkpoint_base = self.data[0]
        # iterate over length of dataset
        self.weights = []
        for idx in tqdm.tqdm(range(self.__len__())):
            checkpoint = copy.deepcopy(self.data[idx])
            ddx = vectorize_checkpoint(checkpoint)
            self.weights.append(ddx)
        # stack to tensor
        self.weights = torch.stack(self.weights, dim=0)
    """


## helper function for property reading
def read_properties_from_path(path, idx, idx_offset):
    """
    reads path/result.json
    returns the dict for training_iteration=idx
    idx_offset=0 if checkpoint_0 was written, else idx_offset=1
    """
    # read json
    try:
        fname = Path(path).joinpath("result.json")
        results = []
        for line in fname.open():
            results.append(json.loads(line))
        # trial_id = results[0]["trial_id"]
    except Exception as e:
        logging.error(f"error loading {fname} - {e}", exc_info=True)
    # pick results
    jdx = idx - idx_offset
    return results[jdx]


############## load_checkpoint_remote ########################################################
@ray.remote(num_returns=4)
def load_checkpoints_remote(
    path,  # path to model
    edx,  # checkpoint_epoch/number
    weight_threshold,
    filter_function,
    verbosity,
    pba,
):
    ## get full path to files ################################################################
    chkpth = path.joinpath(f"checkpoint_{edx}", "checkpoints")
    ## load checkpoint #######################################################################
    chkpoint = {}
    try:
        # try with conventional naming scheme
        try:
            # load chkpoint to cpu memory
            chkpoint = torch.load(str(chkpth), map_location=torch.device("cpu"))
        except FileNotFoundError as e:
            logging.debug(f"File not found. try again with different formatting: {e}")

            # use other formatting
            chkpth = path.joinpath(f"checkpoint_{edx:06d}", "checkpoints")
            # load chkpoint to cpu memory
            chkpoint = torch.load(str(chkpth), map_location=torch.device("cpu"))
    except Exception as e:
        logging.debug(f"error while loading {chkpth}: {e}")
        # instead of appending empty stuff, jump to next
        pba.update.remote(1)
        return None, None, None, None
    ## create label ##########################################################################
    label = f"{path}#_#epoch_{edx}"

    ### check for NAN values #################################################################
    nan_flag = test_checkpoint_for_nan(copy.deepcopy(chkpoint))
    if nan_flag == True:
        if verbosity > 5:
            # jump to next sample
            raise ValueError(f"found nan values in checkpoint {label}")
        pba.update.remote(1)
        return None, None, None, None

    #### apply filter function #################################################################
    if filter_function is not None:
        filter_flag = filter_function(path)
        if filter_flag == True:  # model needs to be filtered
            pba.update.remote(1)
            return None, None, None, None

    #### apply threhold #################################################################
    thresh_flag = test_checkpoint_with_threshold(
        copy.deepcopy(chkpoint), threshold=weight_threshold
    )
    if thresh_flag == True:
        if verbosity > 5:
            # jump to next sample
            raise ValueError(f"found values above threshold in checkpoint {label}")
        pba.update.remote(1)
        return None, None, None, None

    ### clean data #################################################################
    else:  # use data
        ddx = copy.deepcopy(chkpoint)
        ldx = copy.deepcopy(label)

    # return
    pba.update.remote(1)
    return ddx, ldx, path, edx


def load_checkpoint(
    path,  # path to model
    edx,  # checkpoint_epoch/number
    weight_threshold,
    filter_function,
):
    ## get full path to files ################################################################
    chkpth = path.joinpath(f"checkpoint_{edx}", "checkpoints")
    ## load checkpoint #######################################################################
    chkpoint = {}
    try:
        # try with conventional naming scheme
        try:
            # load chkpoint to cpu memory
            chkpoint = torch.load(str(chkpth), map_location=torch.device("cpu"))
        except FileNotFoundError as e:
            logging.debug(f"File not found. try again with different formatting: {e}")

            # use other formatting
            chkpth = path.joinpath(f"checkpoint_{edx:06d}", "checkpoints")
            # load chkpoint to cpu memory
            chkpoint = torch.load(str(chkpth), map_location=torch.device("cpu"))
    except Exception as e:
        logging.debug(f"error while loading {chkpth}: {e}")
        # instead of appending empty stuff, jump to next
        return None, None, None, None
    ## create label ##########################################################################
    label = f"{path}#_#epoch_{edx}"

    ### check for NAN values #################################################################
    nan_flag = test_checkpoint_for_nan(copy.deepcopy(chkpoint))
    if nan_flag == True:
        return None, None, None, None

    #### apply filter function #################################################################
    if filter_function is not None:
        filter_flag = filter_function(path)
        if filter_flag == True:  # model needs to be filtered
            return None, None, None, None

    #### apply threhold #################################################################
    thresh_flag = test_checkpoint_with_threshold(
        copy.deepcopy(chkpoint), threshold=weight_threshold
    )
    if thresh_flag == True:
        return None, None, None, None

    ### clean data #################################################################
    else:  # use data
        ddx = copy.deepcopy(chkpoint)
        ldx = copy.deepcopy(label)

    # retur
    return ddx, ldx, path, edx
