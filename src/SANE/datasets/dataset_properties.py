from pathlib import Path

import torch

from torch.utils.data import Dataset

import random
import copy
import json
import tqdm

import logging


class PropertyDataset(Dataset):
    """
    This dataset class loads model properties from path, and skips the actual checkpoints.
    Interfaces with the same dataset.properties logic of the other datasets, but cheaper to laod and much smaller.
    """

    ## class arguments

    # init
    def __init__(
        self,
        root,  # path from which to load the dataset
        epoch_lst=[5, 10],  # list of epochs to load
        train_val_test="train",  # determines whcih dataset split to use
        ds_split=[0.7, 0.3],  # sets ration between [train, test] or [train, val, test]
        property_keys=None,  # keys of properties to load
        num_threads=4,
        verbosity=0,
    ):
        self.epoch_lst = epoch_lst
        self.verbosity = verbosity
        self.property_keys = copy.deepcopy(property_keys)
        self.train_val_test = train_val_test
        self.ds_split = ds_split

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
        random.seed(42)
        random.shuffle(self.path_list)

        ### Split Train and Test set ###########################################################################
        assert sum(self.ds_split) == 1.0, "dataset splits do not equal to 1"
        # two splits
        if len(self.ds_split) == 2:
            if self.train_val_test == "train":
                idx1 = int(self.ds_split[0] * len(self.path_list))
                self.path_list = self.path_list[:idx1]
            elif self.train_val_test == "test":
                idx1 = int(self.ds_split[0] * len(self.path_list))
                idx2 = idx1 + int(self.ds_split[1] * len(self.path_list))
                self.path_list = self.path_list[idx2:]
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

        ### initialize data over epochs #####################
        if not isinstance(epoch_lst, list):
            epoch_lst = [epoch_lst]

        # ### prepare data lists ###############
        self.paths = []
        # epochs = []

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
                    idx_offset=1,
                )
            except AssertionError as e:
                logging.error(e)
                self.read_properties(
                    results_key_list=result_keys,
                    config_key_list=config_keys,
                    idx_offset=0,
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
        return len(self.data_in)

    ## read properties from path ##############################################################################################################################################
    def read_properties(self, results_key_list, config_key_list, idx_offset=1):
        """
        iterate over all paths in path_list and load the properties
        """
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
        # iterate over samples
        for iidx, ppdx in tqdm.tqdm(enumerate(self.path_list)):
            # iterate over epochs
            for eedx in self.epoch_lst:
                try:
                    res_tmp = read_properties_from_path(
                        ppdx, eedx, idx_offset=idx_offset, verbosity=self.verbosity
                    )
                    if res_tmp is None:
                        continue
                    else:
                        for key in results_key_list:
                            properties[key].append(res_tmp[key])
                        for key in config_key_list:
                            properties[key].append(res_tmp["config"][key])
                        # compute ggap
                        if read_ggap:
                            gap = res_tmp["train_acc"] - res_tmp["test_acc"]
                            properties["ggap"].append(gap)
                        # assert epoch == training_iteration -> match correct data
                        # if iidx == 0:
                        #     train_it = int(res_tmp["training_iteration"])
                        #     assert (
                        #         int(eedx) == train_it
                        #     ), f"training iteration {train_it} and epoch {eedx} don't match."
                    self.paths.append(ppdx)
                except Exception as e:
                    logging.error(e)
                    logging.error(f"couldn't read data from {ppdx}. skip.")
        self.properties = properties


## helper function for property reading
def read_properties_from_path(path, idx, idx_offset, verbosity=5):
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
        if verbosity > 5:
            logging.error(f"error loading {fname}")
            logging.error(e)
    # pick results
    jdx = idx - idx_offset
    try:
        resdx = results[jdx]
        return resdx
    except Exception as e:
        logging.debug(e)
        return None
