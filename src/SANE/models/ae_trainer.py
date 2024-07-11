# from ray.tune.utils import wait_for_gpu
import torch
import sys

import psutil
import os

import json

# print(f"sys path in experiment: {sys.path}")
from pathlib import Path

# import model_definitions
from SANE.models.def_AE_module import AEModule

from torch.utils.data import DataLoader

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


from SANE.models.def_downstream_module import (
    DownstreamTaskLearner as DownstreamTaskLearner,
)
import copy
import wandb


###############################################################################
# define AE_trainer class
###############################################################################
class AE_trainer:
    """
    trainer wrapper around AE model experiments
    Loads datasets, configures model, performs (training) steps and returns performance metrics
    Args:
        config (dict): config dictionary
        data (dict): data dictionary (optional)
    """

    def __init__(self, config):
        """
        Init function to set up experiment. Configures data, augmentaion, and module
        Args:
            config (dict): config dictionary
        """
        logging.info("Set up AE Trainable")
        logging.debug(f"Trainable Config: {config}")

        # set trainable properties
        self.config = config
        self.seed = config["seed"]

        # LOAD DATASETS
        logging.info("get datasets")
        (
            self.trainset,
            self.testset,
            self.valset,
            self.trainloader,
            self.testloader,
            self.valloader,
        ) = self.load_datasets()

        self.config["training::steps_per_epoch"] = len(self.trainloader)

        # SET DOWNSTREAM TASK LEARNERS
        logging.info("set downstream tasks")
        self.load_downstreamtasks()

        # IF RESTORE FROM PREVIOUS CHECKPOINT: LOAD PREVIOUS CONFIG
        if config.get("model::checkpoint_path", None):
            config_path = config.get("model::checkpoint_path", None).joinpath(
                "..", "params.json"
            )
            logging.info(
                f"restore model from previous checkpoint. load config from {config_path}"
            )
            config_old = json.load(config_path.open("r"))
            # transfer all 'model' keys to
            for key in config_old.keys():
                if "model::" in key:
                    self.config[key] = config_old[key]

        # INSTANCIATE MODEL
        logging.info("instanciate model")
        self.module = AEModule(config=config)

        # set trafos
        # GET TRANSFORAMTIONS
        logging.info("set transformations")
        trafo_train, trafo_test, trafo_dst = get_transformations(config)
        self.module.set_transforms(trafo_train, trafo_test, trafo_dst)

        # load checkpoint
        if config.get("model::checkpoint_path", None):
            logging.info(
                f'restore model state from {config.get("model::checkpoint_path",None)}'
            )
            # load all state dicts
            self.load_checkpoint(config.get("model::checkpoint_path", None))
            # reset optimizer
            self.module.set_optimizer(config)

        # SET TRAINABLE CALLBACKS
        logging.info("set callbacks")
        self.callbacks = self.config.get("callbacks", None)

        # run first test epoch and log results
        logging.info("module setup done")
        self._iteration = 0

    def resume(self, iteration):
        # load checkpoint
        logging.info(f"attempting to resume from iteration: {iteration}")
        experiment_dir = self.config["experiment_dir"]
        checkpoint_path = experiment_dir.joinpath(f"checkpoint_{iteration:06d}")

        logging.info(f"loading checkpoint from  {checkpoint_path}")
        self.load_checkpoint(checkpoint_path)
        logging.info(f"checkpoint loaded")
        # set _iteration
        self._iteration = iteration
        #

    # step ####
    def step(self):
        # set model to eval mode as default
        self.module.model.eval()

        # init return
        result_dict = {}

        # perform ssl training task step
        perf_ssl = self.step_ssl()
        # collect metrics
        for key in perf_ssl.keys():
            result_dict[key] = perf_ssl[key]

        # Downstream Taks
        perf_dstk = self.step_dstk()
        # collect metrics
        for key in perf_dstk.keys():
            result_dict[key] = perf_dstk[key]

        # callbacks
        perf_callback = self.step_callback()
        # collect metrics
        for key in perf_callback.keys():
            result_dict[key] = perf_callback[key]

        # monitor memory
        if self.config.get("monitor_memory", False):
            mem_stats = monitor_memory()
            for key in mem_stats.keys():
                result_dict[key] = mem_stats[key]

        return result_dict

    def train(self, resume_iter=0):
        """
        performs training steps and logs results
        """
        experiment_dir = self.config["experiment_dir"]
        json_filename = experiment_dir.joinpath("results.json").absolute()
        config_filename = experiment_dir.joinpath("params.json").absolute()
        # wandb login
        # Read the API key from the text file
        with open(self.config["wandb::api_key_file"], "r") as api_key_file:
            api_key = api_key_file.read().strip()
        # Set the API key as an environment variable
        os.environ["WANDB_API_KEY"] = api_key
        wandb.login()
        # wandb init
        wandb.init(
            project=self.config["wandb::project"],
            dir=experiment_dir,
            config=self.config,
        )
        log_config(config_filename, self.config)
        # resume
        if resume_iter != 0:
            self.resume(iteration=resume_iter)
        # iterate over training steps
        while self._iteration <= self.config["training::epochs_train"]:
            logging.info(f"start iteration {self._iteration}")
            # perform step
            results = self.step()
            # log results
            wandb.log(results)
            # log json results
            update_json_with_results(json_filename, results)
            # save checkpoint
            if self._iteration % self.config["training::output_epoch"] == 0:
                checkpoint_path = experiment_dir.joinpath(
                    f"checkpoint_{self._iteration:06d}"
                )
                self.save_checkpoint(checkpoint_path)
            # log to stdout
            logging.info(f"Iteration {self._iteration} results:")
            logging.info(results)
            # update iterator
            self._iteration += 1

    def step_ssl(
        self,
    ):
        """
        Runs self-supervised training epochs
        """
        result_dict = {}
        # TRAIN EPOCH(s)
        # run several training epochs before one test epoch
        if self._iteration == 0:
            print("test first validation mode")
            perf_train = self.module.test_epoch(
                self.trainloader, epoch=self._iteration, show_progress=True
            )
        else:
            for _ in range(self.config["training::test_epochs"]):
                # set model to training mode
                self.module.model.train()
                # run one training epoch
                perf_train = self.module.train_epoch(
                    self.trainloader, epoch=self._iteration, show_progress=True
                )
                # set model to training mode
                self.module.model.eval()
        # collect metrics
        for key in perf_train.keys():
            result_dict[f"{key}_train"] = perf_train[key]

        # TEST EPOCH
        perf_test = self.module.test_epoch(
            self.testloader, epoch=self._iteration, show_progress=True
        )
        # collect metrics
        for key in perf_test.keys():
            result_dict[f"{key}_test"] = perf_test[key]

        # VALIDATION EPOCH
        if self.valloader is not None:
            # run one test epoch
            perf_val = self.module.test_epoch(
                self.valloader, epoch=self._iteration, show_progress=True
            )
            # collect metrics
            for key in perf_val.keys():
                result_dict[f"{key}_val"] = perf_val[key]

        # return
        return result_dict

    def step_dstk(self):
        """
        runs downstream task evaluation
        """
        result_dict = {}
        # if DownstreamTaskLearner exist. apply downstream task
        # regular dataset
        if self.dstk is not None:
            performance = self.dstk.eval_dstasks(
                # model=self.module.model,
                model=self.module,
                trainset=self.dataset_train_dwst,
                testset=self.dataset_test_dwst,
                valset=self.dataset_val_dwst,
                task_keys=self.task_keys,
                batch_size=self.config["trainset::batchsize"],
            )
            # append performance values to result_dict
            for key in performance.keys():
                new_key = f"dstk/{key}"
                result_dict[new_key] = performance[key]
        # return
        return result_dict

    def step_callback(self):
        """
        runs callbacks at end of epoch
        """
        result_dict = {}
        #
        if self.callbacks:
            logging.info(f"calling on_validation_epoch callback")
            for idx, clbk in enumerate(self.callbacks):
                logging.info(f"callback {idx}")
                # iterations are updated after step, so 1 has to be added.
                perf_callback = clbk.on_validation_epoch_end(
                    self.module, self._iteration + 1
                )
                # collect metrics
                for key in perf_callback.keys():
                    result_dict[key] = perf_callback[key]
        # return
        return result_dict

    def save_checkpoint(self, experiment_dir):
        """
        saves model checkpoint and optimizer state_dict
        Args:
            experiment_dir: path to experiment directory for model saving
        Returns:
            experiment_dir: path to experiment directory for model saving as per tune convention
        """
        self.module.save_model(experiment_dir)
        # tune apparently expects to return the directory
        return experiment_dir

    def load_checkpoint(self, experiment_dir):
        """
        loads model checkpoint and optimizer state_dict
        Uses self.reset_optimizer to decide if optimizer should be loaded
        Args:
            experiment_dir: path to experiment directory for model loading
        Returns:
            experiment_dir: path to experiment directory for model loading as per tune convention
        """
        self.module.load_model(experiment_dir)
        # TODO: re-compile model if necessary.
        # tune apparently expects to return the directory
        return experiment_dir

    def load_datasets(self):
        if "dataset.pt" in str(self.config["dataset::dump"]):
            # import conventional downstream module
            windowsize = self.config.get("training::windowsize", 15)
            # MULTIWINDOW
            if self.config.get("trainset::multi_windows", None):
                trafo_dataset = MultiWindowCutter(
                    windowsize=windowsize, k=self.config.get("trainset::multi_windows")
                )
            # init dataloaders
            logging.info("Load Data")
            # load dataset from file
            dataset = torch.load(self.config["dataset::dump"])

            trainset = dataset["trainset"]
            testset = dataset["testset"]
            valset = dataset.get("valset", None)

            # transfer trafo_dataset to datasets
            trainset.transforms = trafo_dataset  # this applies multi-windowcutter, etc.
            testset.transforms = trafo_dataset
            if valset is not None:
                valset.transforms = trafo_dataset

            # get full dataset in tensors
            logging.info("set up dataloaders")
            # correct dataloader batchsize with # of multi_window samples out of single __getitem__ call
            assert (
                self.config["trainset::batchsize"]
                % self.config.get("trainset::multi_windows", 1)
                == 0
            ), f'batchsize {self.config["trainset::batchsize"]} needs to be divisible by multi_windows {self.config["trainset::multi_windows"]}'
            bs_corr = int(
                self.config["trainset::batchsize"]
                / self.config.get("trainset::multi_windows", 1)
            )
            logging.info(f"corrected batchsize to {bs_corr}")
            #
            trainloader = DataLoader(
                trainset,
                batch_size=bs_corr,
                shuffle=True,
                drop_last=True,  # important: we need equal batch sizes
                num_workers=self.config.get("trainloader::workers", 2),
                prefetch_factor=4,
            )

            # get full dataset in tensors
            testloader = torch.utils.data.DataLoader(
                testset,
                batch_size=bs_corr,
                shuffle=False,
                drop_last=True,  # important: we need equal batch sizes
                num_workers=self.config.get("testloader::workers", 2),
                prefetch_factor=4,
            )
            if valset is not None:
                # get full dataset in tensors
                valloader = torch.utils.data.DataLoader(
                    valset,
                    batch_size=bs_corr,
                    shuffle=False,
                    drop_last=True,  # important: we need equal batch sizes
                    num_workers=self.config.get("testloader::workers", 2),
                    prefetch_factor=4,
                )
            else:
                valloader = None

            return trainset, testset, valset, trainloader, testloader, valloader
        else:
            raise NotImplementedError(
                f'could not load dataset from {self.config["dataset::dump"]}'
            )

    def load_downstreamtasks(self):
        """
        load downstream task datasets and instanciate downstream task learner
        """
        if self.config.get("downstreamtask::dataset", None):
            # load datasets
            downstream_dataset_path = self.config.get("downstreamtask::dataset", None)
            # conventional dataset
            if "dataset.pt" in str(downstream_dataset_path):
                # load conventional pickeld dataset
                pth_tmp = str(downstream_dataset_path).replace(
                    "dataset.pt", "dataset_train.pt"
                )
                self.dataset_train_dwst = torch.load(pth_tmp)
                pth_tmp = str(downstream_dataset_path).replace(
                    "dataset.pt", "dataset_test.pt"
                )
                self.dataset_test_dwst = torch.load(pth_tmp)
                pth_tmp = str(downstream_dataset_path).replace(
                    "dataset.pt", "dataset_val.pt"
                )
                self.dataset_val_dwst = torch.load(pth_tmp)

                # instanciate downstreamtask module
                if self.dataset_train_dwst.properties is not None:
                    logging.info(
                        "Found properties in dataset - downstream tasks are going to be evaluated at test time."
                    )
                    self.dstk = DownstreamTaskLearner()

                    dataset_info_path = str(downstream_dataset_path).replace(
                        "dataset.pt", "dataset_info_test.json"
                    )
        elif (
            self.config.get("downstreamtask::dataset", "use_ssl_dataset")
            == "use_ssl_dataset"
        ):
            # fallback to training dataset for downstream tasks
            if "dataset.pt" in str(self.config["dataset::dump"]):
                if self.trainset.properties is not None:
                    # assign correct datasets
                    self.dataset_train_dwst = self.trainset
                    self.dataset_test_dwst = self.testset
                    self.dataset_val_dwst = self.valset
                    logging.info(
                        "Found properties in dataset - downstream tasks are going to be evaluated at test time."
                    )
                    self.dstk = DownstreamTaskLearner()

                    dataset_info_path = str(self.config["dataset::dump"]).replace(
                        "dataset.pt", "dataset_info_test.json"
                    )


        else:
            logging.info("No properties found in dataset - skip downstream tasks.")
            self.dstk = None

        # load task_keys
        if self.dstk:
            try:
                self.dataset_info = json.load(open(dataset_info_path, "r"))
                self.task_keys = self.dataset_info["properties"]
            except Exception as e:
                print(e)
                self.task_keys = [
                    "test_acc",
                    "training_iteration",
                    "ggap",
                ]


def get_transformations(config):
    """
    get transformations for training, testing, and downstream task
    Args:
        config (dict): config dictionary
    Returns:
        trafo_train (AugmentationPipeline): augmentation pipeline for training
        trafo_test (AugmentationPipeline): augmentation pipeline for testing
        trafo_dst (AugmentationPipeline): augmentation pipeline for downstream task
    """
    # set windowsize
    windowsize = config.get("training::windowsize", 15)
    # TRAIN AUGMENTATIONS
    stack_1 = []
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

    # downstream task permutation (chose which permutationn to use for dstk)
    if config.get("training::permutation_number", 0) > 0:
        trafo_dst = PermutationSelector(mode="canonical", keep_properties=True)
    else:
        trafo_dst = PermutationSelector(mode="identity", keep_properties=True)

    return trafo_train, trafo_test, trafo_dst


def monitor_memory():
    # identify current process
    current_process = psutil.Process(os.getpid())
    # get the current memory usage
    mem_main = current_process.memory_info().rss
    logging.info(f"memory usage - main: {mem_main / 1024**3} GB")
    # get memory usage for all child processes
    mem_tot = mem_main
    for child in current_process.children(recursive=True):
        mem_tot += child.memory_info().rss

    logging.info(f"memory usage - total: {mem_tot / 1024**3} GB")
    out = {
        "memory_usage_main": mem_main / 1024**3,
        "memory_usage_total": mem_tot / 1024**3,
    }
    return out


def update_json_with_results(json_filename, new_dict):
    # curtosy to chatgpt
    # clean repo
    out_dict = copy.deepcopy(new_dict)
    for key in out_dict.keys():
        if isinstance(out_dict[key], Path):
            # cast to str
            out_dict[key] = str(out_dict[key].absolute())
    # Check if the JSON file already exists
    if os.path.exists(json_filename):
        with open(json_filename, "r") as json_file:
            training_dict = json.load(json_file)
    else:
        training_dict = []
    # Append the new dict to the existing list of training_dict
    training_dict.append(out_dict)
    # Write the updated list back to the JSON file
    with open(json_filename, "w") as json_file:
        json.dump(training_dict, json_file, indent=4)


def log_config(config_filename, config):
    # clean repo
    out_config = copy.deepcopy(config)
    out_config.pop("callbacks", None)
    for key in out_config.keys():
        if isinstance(out_config[key], Path):
            # cast to str
            out_config[key] = str(out_config[key].absolute())
    # log config
    with config_filename.open("w") as f:
        json.dump(out_config, f)
