#########
# imports
#########
# set environment variables to limit cpu usage
import os

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import json
import pandas as pd
import tqdm

import umap

import random

import ray
from ray import tune

import pingouin as pg
from sklearn.neighbors import KernelDensity
import scipy.stats as st

from SANE.datasets.dataset_auxiliaries import vector_to_checkpoint
from SANE.datasets.dataset_base import ModelDatasetBase
from SANE.datasets.dataset_properties import PropertyDataset

from SANE.models.def_net import NNmodule
from SANE.models.def_NN_experiment_from_checkpoint import (
    NN_tune_trainable_from_checkpoint,
)

# from SANE.models.def_simclr_ae_module import SimCLRAEModule


########################################################################
# # find domain from path ###############################################
# #######################################################################
def find_domain_from_path(population_path: Path):
    pdx = str(population_path)
    domain = ""
    if "/mnist" in pdx:
        domain = "MNIST"
    if "/svhn" in pdx:
        domain = "SVHN"
    if "/cifar10" in pdx:
        domain = "CIFAR-10"
    if "/stl10" in pdx:
        domain = "STL-10"
    return domain


########################################################################
# # SAMPLE WEIGHTS ######################################################
# #######################################################################
def sample(
    experiment_path: Path,
    path_to_samples: Path = Path("."),
    no_samples: int = 150,
):
    """
    experiment_path: pathlib.Path to experiment directory of representation learner
    path_to_samples: pathlib.Path directory in which the sampled checkpoints will be dropped
    no_samples: int number of samples to be drawn
    """
    ########################
    ## LOAD MODEL AND DATA
    ########################
    print(f"########### load model and data")
    # load data
    data_path = experiment_path.joinpath("dataset.pt")
    dataset = torch.load(data_path.absolute())

    model_config_path = experiment_path.joinpath("config_zoo.json")
    # load hyper-representation model
    # config
    model_path = experiment_path
    config_path = model_path.joinpath("config_ae.json")
    config = json.load(config_path.open("r"))
    # device
    config["device"] = "cpu"
    config["model::type"] = "transformer"
    lat_size = config["model::latent_dim"]
    # load checkpoint
    checkpoint_path = model_path.joinpath(f"checkpoint_ae.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if no_samples > 0:
        # get weights
        weights_train = dataset["trainset"].__get_weights__()
        # get epochs
        epochs_train = dataset["trainset"].epochs

        ########################
        ## Basline - Finetune
        ########################
        print(f"########### Baseline: B_F")
        # get max epoch
        epoch_last = torch.tensor(epochs_train).max().item()  # avoid max(list)
        # slice weights for last epoch
        idx_last_epoch = [
            idx for idx, edx in enumerate(epochs_train) if edx == epoch_last
        ]
        weights_transfer = weights_train[idx_last_epoch, :]
        # slice for no_samples
        weights_transfer = weights_transfer[:no_samples, :]
        # save checkpoints
        generate_checkpoints_from_weights(
            sample_population_path=path_to_samples.joinpath("direct"),
            weights=weights_transfer,
            model_config_path=model_config_path,
            layer_lst=config["trainset::layer_lst"],
        )

        # init model
        print(f"Load Hyper-Representation Model")

        AE = SimCLRAEModule(config)
        # load state_dict
        AE.model.load_state_dict(checkpoint)
        # put model in eval mode
        AE.model.eval()

        path_to_samples.mkdir(exist_ok=True)
        ########################
        ## SAMPLE 1: UNIFORM
        ########################
        print(f"########### sample 1: uniform")
        sample_1 = torch.rand(size=[no_samples, lat_size], requires_grad=False) * 2 - 1
        # forward pass through decoder
        with torch.no_grad():
            weights_1 = AE.forward_decoder(sample_1)
        # save checkpoints
        generate_checkpoints_from_weights(
            sample_population_path=path_to_samples.joinpath("uniform"),
            weights=weights_1,
            model_config_path=model_config_path,
            layer_lst=config["trainset::layer_lst"],
        )

        ########################
        ## SAMPLE 2: TRAIN
        ########################
        print(f"########### sample 2: B_T")
        # Sample 2: full trainset -> mask sample in low dim space
        ## map train weights to embeddings

        with torch.no_grad():
            z_train = AE.forward_encoder(weights_train)
        # fit UMAP to train set
        umap_reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.01,
            n_components=3,
            metric="euclidean",
            random_state=42,
        )
        # get lower-dim representation
        z_train_umap = umap_reducer.fit_transform(z_train.detach().numpy())
        # sample in umap range
        min = z_train_umap.min()
        max = z_train_umap.max()
        sample_2 = (
            torch.rand(size=[no_samples, 3], requires_grad=False) * (max - min) + min
        )
        # reconstruct
        z_umap_sample_2 = umap_reducer.inverse_transform(sample_2)
        z_umap_sample_2 = torch.Tensor(z_umap_sample_2)

        with torch.no_grad():
            weights_2 = AE.forward_decoder(z_umap_sample_2)
        # save checkpoints
        generate_checkpoints_from_weights(
            sample_population_path=path_to_samples.joinpath("train"),
            weights=weights_2,
            model_config_path=model_config_path,
            layer_lst=config["trainset::layer_lst"],
        )

        ########################
        ## SAMPLE 3: TRAIN_best
        ########################
        print(f"########### sample 3: B_T30")
        # Sample 2: best 30% of trainset -> mask sample in low dim space
        train_zoo_acc = torch.Tensor(dataset["trainset"].properties["test_acc"])
        acc_threshold = torch.quantile(train_zoo_acc, q=0.7)
        idx_best = [
            idx
            for idx, acc_dx in enumerate(dataset["trainset"].properties["test_acc"])
            if acc_dx >= acc_threshold
        ]
        ## map train weights to embeddings
        with torch.no_grad():
            z_train_best = AE.forward_encoder(weights_train[idx_best])
        # fit UMAP to train set
        umap_reducer_best = umap.UMAP(
            n_neighbors=15,
            min_dist=0.01,
            n_components=3,
            metric="euclidean",
            random_state=42,
        )
        # get lower-dim representation
        z_train_best_umap = umap_reducer_best.fit_transform(
            z_train_best.detach().numpy()
        )
        # sample in umap range
        min = z_train_best_umap.min()
        max = z_train_best_umap.max()
        sample_3 = (
            torch.rand(size=[no_samples, 3], requires_grad=False) * (max - min) + min
        )
        # reconstruct
        z_umap_sample_3 = umap_reducer_best.inverse_transform(sample_3)
        z_umap_sample_3 = torch.Tensor(z_umap_sample_3)

        with torch.no_grad():
            weights_3 = AE.forward_decoder(z_umap_sample_3)
        # save checkpoints
        generate_checkpoints_from_weights(
            sample_population_path=path_to_samples.joinpath("best"),
            weights=weights_3,
            model_config_path=model_config_path,
            layer_lst=config["trainset::layer_lst"],
        )

        ########################
        ## SAMPLE 4/5: TRAIN_z - best z
        ########################
        ### draw samples from z-wise distributions
        print(f"########### sample 4/5: S_KDE/30")

        kde = KernelDensity(kernel="gaussian", bandwidth=0.002)
        z_samples_full = []
        z_samples_top30 = []
        for idx in range(z_train.shape[1]):
            # case 1: all
            z_full = z_train[:, idx].unsqueeze(dim=1)
            kde.fit(z_full)
            z_full_tmp = kde.sample(n_samples=no_samples, random_state=42)
            z_samples_full.append(torch.tensor(z_full_tmp))

            # case 2: top30
            z_top30 = z_full[idx_best, :]
            kde.fit(z_top30)
            z_top30_tmp = kde.sample(n_samples=no_samples, random_state=42)
            z_samples_top30.append(torch.tensor(z_top30_tmp))
        # print(len(z_samples_full))
        # print(z_samples_full[0].shape)
        z_samples_full = torch.cat(z_samples_full, dim=1).float()
        z_samples_top30 = torch.cat(z_samples_top30, dim=1).float()
        # print(z_samples_full.shape)
        # print(z_samples_top30.shape)
        with torch.no_grad():
            weights_z_samples_full = AE.forward_decoder(z_samples_full)
            weights_z_samples_top30 = AE.forward_decoder(z_samples_top30)
        # save checkpoints
        generate_checkpoints_from_weights(
            sample_population_path=path_to_samples.joinpath("kde_z_train"),
            weights=weights_z_samples_full,
            model_config_path=model_config_path,
            layer_lst=config["trainset::layer_lst"],
        )
        generate_checkpoints_from_weights(
            sample_population_path=path_to_samples.joinpath("kde_z_best"),
            weights=weights_z_samples_top30,
            model_config_path=model_config_path,
            layer_lst=config["trainset::layer_lst"],
        )

    return config_path, model_config_path


########################################################################
# # SAMPLE WEIGHTS ######################################################
# #######################################################################
def generate_checkpoints_from_weights(
    sample_population_path, weights, model_config_path, layer_lst
):
    # generate path
    sample_population_path = Path(sample_population_path)
    sample_population_path.mkdir(exist_ok=True)
    # load config
    model_config = json.load(model_config_path.open("r"))
    # init model
    base_model = NNmodule(model_config)
    checkpoint_base = base_model.model.state_dict()
    # iterate over samples
    for idx in tqdm.tqdm(range(weights.shape[0])):
        # slice
        weight_vector = weights[idx, :].clone()
        # checkpoint
        chkpt = vector_to_checkpoint(
            checkpoint=checkpoint_base,
            vector=weight_vector,
            layer_lst=layer_lst,
            use_bias=True,
        )
        # load state_dict - check for errors
        base_model.model.load_state_dict(chkpt)
        #
        fname = sample_population_path.joinpath(f"checkpoint_{idx}.pt")
        torch.save(chkpt, fname)


########################################################################
# # FINETUNE MODELS TO TARGET DOMAIN ####################################
# #######################################################################
def finetune(
    project: str,
    population_path: Path,
    path_to_samples: Path,
    path_target_zoo: Path,
    model_config_path: Path,
    model_config: str,
    no_samples: int,
    training_epochs: int = 25,
    cpus=6,
    cpu_per_trial: int = 1,
    skip=None,
):
    """
    project: str with experiment project, .e.g "MNIST_to_SVHN"
    population_path: Path directory containing the finetuned / transferred populations. should be unique for each combination (contain source/target names)
    path_to_samples: Path directory containing the sample checkpoints
    path_target_zoo: Path directory containing the target/id population
    model_config_path: Path path to small model config
    model_config: either ["source","target"] depending on from where the config is supposed to be taken
    no_samples: int number of samples to finetune
    """
    ### finetune direct
    if "direct" not in skip:
        finetune_single(
            project=project,
            group="direct",
            path_to_samples=path_to_samples.joinpath("direct"),
            population_path=population_path,
            model_config_path=model_config_path,
            target_zoo_path=path_target_zoo,
            model_config=model_config,
            no_samples=no_samples,
            sample_epoch=None,
            training_epochs=training_epochs,
            cpus=cpus,
            cpu_per_trial=cpu_per_trial,
        )

    ### finetune best
    if "best" not in skip:
        finetune_single(
            project=project,
            group="best",
            path_to_samples=path_to_samples.joinpath("best"),
            population_path=population_path,
            model_config_path=model_config_path,
            target_zoo_path=path_target_zoo,
            model_config=model_config,
            no_samples=no_samples,
            sample_epoch=None,
            training_epochs=training_epochs,
            cpus=cpus,
            cpu_per_trial=cpu_per_trial,
        )

    ### finetune best
    if "kde_z_best" not in skip:
        finetune_single(
            project=project,
            group="kde_z_best",
            path_to_samples=path_to_samples.joinpath("kde_z_best"),
            population_path=population_path,
            model_config_path=model_config_path,
            target_zoo_path=path_target_zoo,
            model_config=model_config,
            no_samples=no_samples,
            sample_epoch=None,
            training_epochs=training_epochs,
            cpus=cpus,
            cpu_per_trial=cpu_per_trial,
        )

    ### finetune kde_z_train
    if "kde_z_train" not in skip:
        finetune_single(
            project=project,
            group="kde_z_train",
            path_to_samples=path_to_samples.joinpath("kde_z_train"),
            population_path=population_path,
            model_config_path=model_config_path,
            target_zoo_path=path_target_zoo,
            model_config=model_config,
            no_samples=no_samples,
            sample_epoch=None,
            training_epochs=training_epochs,
            cpus=cpus,
            cpu_per_trial=cpu_per_trial,
        )

    ### finetune train
    if "train" not in skip:
        finetune_single(
            project=project,
            group="train",
            path_to_samples=path_to_samples.joinpath("train"),
            population_path=population_path,
            model_config_path=model_config_path,
            target_zoo_path=path_target_zoo,
            model_config=model_config,
            no_samples=no_samples,
            sample_epoch=None,
            training_epochs=training_epochs,
            cpus=cpus,
            cpu_per_trial=cpu_per_trial,
        )

    ### finetune uniform
    if "uniform" not in skip:
        finetune_single(
            project=project,
            group="uniform",
            path_to_samples=path_to_samples.joinpath("uniform"),
            population_path=population_path,
            model_config_path=model_config_path,
            target_zoo_path=path_target_zoo,
            model_config=model_config,
            no_samples=no_samples,
            sample_epoch=None,
            training_epochs=training_epochs,
            cpus=cpus,
            cpu_per_trial=cpu_per_trial,
        )

    ### finetune uniform
    if "gan" not in skip:
        finetune_single(
            project=project,
            group="gan",
            path_to_samples=path_to_samples.joinpath("gan"),
            population_path=population_path,
            model_config_path=model_config_path,
            target_zoo_path=path_target_zoo,
            model_config=model_config,
            no_samples=no_samples,
            sample_epoch=None,
            training_epochs=training_epochs,
            cpus=cpus,
            cpu_per_trial=cpu_per_trial,
        )

    ### finetune uniform
    if "gan_best" not in skip:
        finetune_single(
            project=project,
            group="gan_best",
            path_to_samples=path_to_samples.joinpath("gan_best"),
            population_path=population_path,
            model_config_path=model_config_path,
            target_zoo_path=path_target_zoo,
            model_config=model_config,
            no_samples=no_samples,
            sample_epoch=None,
            training_epochs=training_epochs,
            cpus=cpus,
            cpu_per_trial=cpu_per_trial,
        )


########################################################################
# # FINETUNE TO SINGLE TARGET DOMAIN ####################################
# #######################################################################
def finetune_single(
    project: str,
    group: str,
    path_to_samples: Path,
    population_path: Path,
    model_config_path: Path,
    target_zoo_path: Path,
    model_config: str = "source",
    no_samples: int = 150,
    sample_epoch: int = None,
    training_epochs: int = 25,
    cpus: int = 6,
    cpu_per_trial: int = 1,
):
    """
    project: str name of project, i.e. 'MNIST_to_SVHN'
    group: str name for the population to be trained, e.g., 'direct' or 'best'
    path_to_samples: Path directory containing the sample checkpoints
    population_path: Path directory containing the finetuned / transferred populations. should be unique for each combination (contain source/target names)
    model_config_path: Path path to small model config
    path_target_zoo: Path directory containing the target/id population
    no_samples: int number of samples to finetune
    sample_epoch: int reference epoch to consider for direct finetuning
    """
    print(f"Finetune: {group}")
    cpus = cpus
    gpus = 0

    gpu_fraction = ((gpus * 100) // (cpus / cpu_per_trial)) / 100
    resources_per_trial = {"cpu": cpu_per_trial, "gpu": gpu_fraction}

    # experiment name
    project = project
    group = group
    experiment_name = group

    # set module parameters
    if model_config == "source":
        config = json.load(model_config_path.open("r"))
    elif model_config == "target":
        config_path = target_zoo_path.joinpath("config_zoo.json")
        config = json.load(config_path.open("r"))
    else:
        raise NotImplementedError

    config["training::epochs_train"] = training_epochs

    if not sample_epoch:
        ## intepret path_to_samples as usual samples of the format "checkpoint_{idx}.pt"
        # set path to load sampled weights from
        config["training::init_checkpoint_path"] = path_to_samples.absolute()
        sample_numbers = list(range(no_samples))
        config["training::sample_number"] = tune.grid_search(sample_numbers)
    else:
        # assume we're loading from a regular zoo now.
        config["training::sample_epoch"] = sample_epoch
        path_list = [f for f in Path(path_to_samples).iterdir() if f.is_dir()]
        # slice for no_samples
        random.seed(42)
        random.shuffle(path_list)
        path_list = path_list[:no_samples]
        config["training::sample_number"] = None
        config["training::init_checkpoint_path"] = tune.grid_search(path_list)

    # set training parameters
    net_dir = population_path
    net_dir.mkdir(parents=True, exist_ok=True)

    print(f"Zoo directory: {net_dir.absolute()}")
    # check if htere are directories in side
    resume = False
    net_dir.joinpath(experiment_name).mkdir(exist_ok=True)
    pth_lst_tmp = [f for f in net_dir.joinpath(experiment_name).iterdir() if f.is_dir()]
    if len(pth_lst_tmp) > 0:
        print(
            f"path {net_dir.joinpath(experiment_name)} not empty found {len(pth_lst_tmp)}. Will attempt to resume trial"
        )
        resume = True

    # load data
    data_path = target_zoo_path.joinpath("dataset.pt").absolute()
    print(f"Data directory: {data_path.absolute()}")
    # load existing dataset
    dataset = torch.load(str(data_path))

    # save datasets in zoo directory
    config["dataset::dump"] = net_dir.joinpath(experiment_name, "dataset.pt").absolute()
    torch.save(dataset, config["dataset::dump"])

    ray.init(
        num_cpus=cpus,
        num_gpus=gpus,
    )

    # save config as json file
    with open((net_dir.joinpath(experiment_name, "config.json")), "w") as f:
        json.dump(config, f, default=str)

    assert ray.is_initialized() == True

    # run tune trainable experiment
    try:
        print(f"train population in {net_dir}")

        experiment = ray.tune.Experiment(
            name=experiment_name,
            run=NN_tune_trainable_from_checkpoint,
            stop={
                "training_iteration": config["training::epochs_train"],
            },
            checkpoint_config=ray.air.CheckpointConfig(
                num_to_keep=None,
                checkpoint_frequency=1,
                checkpoint_at_end=False,
            ),
            config=config,
            local_dir=net_dir,
            resources_per_trial=resources_per_trial,
        )
        # run
        ray.tune.run_experiments(
            experiments=experiment,
            resume=resume,  # resumes from previous run. if run should be done all over, set resume=False
            reuse_actors=False,
            verbose=3,
        )

        # tuner = tune.tuner.Tuner(
        #     NN_tune_trainable_from_checkpoint,
        #     param_space=config,
        #     tune_config=tune.TuneConfig(
        #         # num_samples=1,
        #         metric="test_acc",
        #         mode="max",
        #         max_concurrent_trials=10,
        #     ),
        #     run_config=ray.air.RunConfig(
        #         name=experiment_name,
        #         local_dir=net_dir,
        #         stop={
        #             "training_iteration": config["training::epochs_train"],
        #         },
        #         checkpoint_config=ray.air.CheckpointConfig(
        #             num_to_keep=None,
        #             checkpoint_frequency=1,
        #             checkpoint_at_end=False,
        #         ),
        #         verbose=3,
        #     ),
        # )
        # result_grid = tuner.fit()
        # best_result = result_grid.get_best_result()
        # analysis = tune.run(
        #     reuse_actors=False,
        #     resume=resume,  # resumes from previous run. if run should be done all over, set resume=False
        #     resources_per_trial=resources_per_trial,
        # )
    except Exception as e:
        print(e)

    # If resume: there may be errored trials. Try to reset them and have another go.
    if resume == True:
        resume = "ERRORED_ONLY"
        try:
            ray.tune.run_experiments(
                experiments=experiment,
                resume=resume,  # resumes from previous run. if run should be done all over, set resume=False
                reuse_actors=False,
                verbose=3,
            )
            # print(f"train errored trials in {net_dir}")
            # analysis = tune.run(
            #     NN_tune_trainable_from_checkpoint,
            #     name=experiment_name,
            #     stop={
            #         "training_iteration": config["training::epochs_train"],
            #     },
            #     checkpoint_score_attr="test_acc",
            #     checkpoint_freq=config["training::output_epoch"],
            #     config=config,
            #     local_dir=net_dir,
            #     reuse_actors=False,
            #     resume=resume,  # resumes from previous run. if run should be done all over, set resume=False
            #     resources_per_trial=resources_per_trial,
            #     verbose=3,
            # )
        except Exception as e:
            print(e)

    try:
        assert ray.is_initialized() == False
    except:
        ray.shutdown()

    # clean_up empty / errored only runs
    print(f"clean up {net_dir.joinpath(experiment_name)}: remove uneccessary folders")
    path_lst = [
        pdx for pdx in net_dir.joinpath(experiment_name).iterdir() if pdx.is_dir()
    ]
    # find directories that are either empty or contain error.txt
    path_lst_remove = [
        pdx
        for pdx in path_lst
        if pdx.joinpath("error.txt").is_file() or not any(pdx.iterdir())
    ]
    for pdx in tqdm.tqdm(path_lst_remove):
        rm_tree(pdx)


# #######################################################################
# # PLOT  ###############################################################
# #######################################################################


def rm_tree(pth):
    pth = Path(pth)
    for child in pth.glob("*"):
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child)
    pth.rmdir()


# #######################################################################
# # PLOT  ###############################################################
# #######################################################################
def plot_populations(
    source: str,
    target: str,
    path_target_zoo: Path,
    population_path: Path,
    layer_lst: list,
    id_offset: int = 25,
    plot_domains: list() = [],
):
    ## load zoos
    # target

    result_key_list = [
        "train_loss",
        "test_loss",
        "train_acc",
        "test_acc",
        "training_iteration",
        "trial_id",
    ]
    config_key_list = ["model::nlin", "model::init_type"]
    property_keys = {
        "result_keys": result_key_list,
        "config_keys": config_key_list,
    }

    # set which epochs to load
    epoch_lst_transfer = (
        list(range(0, 51)) if id_offset == 0 else list(range(0, 51 - id_offset))
    )

    print(f"load populations for plotting: 1/9")
    root_path = path_target_zoo.absolute().joinpath("zoo_testsplit.pt")
    dataset_baseline = torch.load(root_path)  # load pre-computed dataset

    # direct transfer
    print(f"load populations for plotting: 2/9")
    root_path = population_path.joinpath("direct").absolute()
    root_path.mkdir(exist_ok=True)  # to catch path_not_exist error
    dataset_direct = PropertyDataset(
        root=[
            root_path,
        ],
        epoch_lst=epoch_lst_transfer,
        train_val_test="train",
        ds_split=[1.0, 0.0],
        property_keys=property_keys,
        verbosity=0,
    )

    # sample 1
    print(f"load populations for plotting: 3/9")
    root_path = population_path.joinpath("uniform").absolute()
    root_path.mkdir(exist_ok=True)  # to catch path_not_exist error
    dataset_uniform = PropertyDataset(
        root=[
            root_path,
        ],
        epoch_lst=epoch_lst_transfer,
        train_val_test="train",
        ds_split=[1.0, 0.0],
        property_keys=property_keys,
        verbosity=0,
    )

    # sample 2
    print(f"load populations for plotting: 4/9")
    root_path = population_path.joinpath("train").absolute()
    root_path.mkdir(exist_ok=True)  # to catch path_not_exist error
    dataset_train = PropertyDataset(
        root=[
            root_path,
        ],
        epoch_lst=epoch_lst_transfer,
        train_val_test="train",
        ds_split=[1.0, 0.0],
        property_keys=property_keys,
        verbosity=0,
    )

    # sample 3
    print(f"load populations for plotting: 5/9")
    root_path = population_path.joinpath("best").absolute()
    root_path.mkdir(exist_ok=True)  # to catch path_not_exist error
    dataset_best = PropertyDataset(
        root=[
            root_path,
        ],
        epoch_lst=epoch_lst_transfer,
        train_val_test="train",
        ds_split=[1.0, 0.0],
        property_keys=property_keys,
        verbosity=0,
    )
    # sample kde_z_train
    print(f"load populations for plotting: 6/9")
    root_path = population_path.joinpath("kde_z_train").absolute()
    root_path.mkdir(exist_ok=True)  # to catch path_not_exist error
    dataset_kde_z_train = PropertyDataset(
        root=[
            root_path,
        ],
        epoch_lst=epoch_lst_transfer,
        train_val_test="train",
        ds_split=[1.0, 0.0],
        property_keys=property_keys,
        verbosity=0,
    )
    # sample 3
    print(f"load populations for plotting: 7/9")
    root_path = population_path.joinpath("kde_z_best").absolute()
    root_path.mkdir(exist_ok=True)  # to catch path_not_exist error
    dataset_kde_z_best = PropertyDataset(
        root=[
            root_path,
        ],
        epoch_lst=epoch_lst_transfer,
        train_val_test="train",
        ds_split=[1.0, 0.0],
        property_keys=property_keys,
        verbosity=0,
    )

    # sample gan
    print(f"load populations for plotting: 8/9")
    root_path = population_path.joinpath("gan").absolute()
    root_path.mkdir(exist_ok=True)  # to catch path_not_exist error
    dataset_gan = PropertyDataset(
        root=[
            root_path,
        ],
        epoch_lst=epoch_lst_transfer,
        train_val_test="train",
        ds_split=[1.0, 0.0],
        property_keys=property_keys,
        verbosity=0,
    )

    # sample gan_best
    print(f"load populations for plotting: 9/9")
    root_path = population_path.joinpath("gan_best").absolute()
    root_path.mkdir(exist_ok=True)  # to catch path_not_exist error
    dataset_gan_best = PropertyDataset(
        root=[
            root_path,
        ],
        epoch_lst=epoch_lst_transfer,
        train_val_test="train",
        ds_split=[1.0, 0.0],
        property_keys=property_keys,
        verbosity=0,
    )

    ### get data
    if source == target:
        epoch_offset = id_offset
    else:
        epoch_offset = 0

    # baseline
    try:
        hue_baseline = [
            # r"$B_{T}$: trained on " + f"{target}"
            r"$B_{T}$"
            for _ in range(len(dataset_baseline.properties["test_acc"]))
        ]
        epochs_baseline = dataset_baseline.properties["training_iteration"]
        acc_baseline = dataset_baseline.properties["test_acc"]
    except:
        hue_baseline = []
        epochs_baseline = []
        acc_baseline = []

    # direct
    try:
        hue_direct = [
            # r"$B_{F}$: pretrained on " + f"{source} and finetuned on {target}"
            r"$B_{F}$"
            for _ in range(len(dataset_direct.properties["test_acc"]))
        ]
        epochs_direct = [
            edx + epoch_offset
            for edx in dataset_direct.properties["training_iteration"]
        ]
        acc_direct = dataset_direct.properties["test_acc"]
    except:
        hue_direct = []
        epochs_direct = []
        acc_direct = []

    # uniform
    try:
        hue_uniform = [
            # r"$S_{0}$: sampled uniform in latent"
            r"$S_{0}$"
            for _ in range(len(dataset_uniform.properties["test_acc"]))
        ]
        epochs_uniform = [
            edx + epoch_offset
            for edx in dataset_uniform.properties["training_iteration"]
        ]
        acc_uniform = dataset_uniform.properties["test_acc"]
    except:
        hue_uniform = []
        epochs_uniform = []
        acc_uniform = []

    # train
    try:
        hue_train = [
            # r"$S_{N}$: neigborhood-based sampling"
            r"$S_{N}$"
            for _ in range(len(dataset_train.properties["test_acc"]))
        ]
        epochs_train = [
            edx + epoch_offset for edx in dataset_train.properties["training_iteration"]
        ]
        acc_train = dataset_train.properties["test_acc"]
    except:
        hue_train = []
        epochs_train = []
        acc_train = []

    # best
    try:
        hue_best = [
            # r"$S_{N30}$: top 30% neighborhood-based sampling"
            r"$S_{N30}$"
            for _ in range(len(dataset_best.properties["test_acc"]))
        ]
        epochs_best = [
            edx + epoch_offset for edx in dataset_best.properties["training_iteration"]
        ]
        acc_best = dataset_best.properties["test_acc"]
    except:
        hue_best = []
        epochs_best = []
        acc_best = []

    # kde train
    try:
        hue_kde_z_train = [
            # r"$S_D$: sampled in z-distribution"
            r"$S_D$"
            for _ in range(len(dataset_kde_z_train.properties["test_acc"]))
        ]
        epochs_kde_z_train = [
            edx + epoch_offset
            for edx in dataset_kde_z_train.properties["training_iteration"]
        ]
        acc_kde_z_train = dataset_kde_z_train.properties["test_acc"]
    except:
        hue_kde_z_train = []
        epochs_kde_z_train = []
        acc_kde_z_train = []

    # kde best
    try:
        hue_kde_z_best = [
            # r"$S_{D30}$: sampled in top 30% z-distribution"
            r"$S_{D30}$"
            for _ in range(len(dataset_kde_z_best.properties["test_acc"]))
        ]
        epochs_kde_z_best = [
            edx + epoch_offset
            for edx in dataset_kde_z_best.properties["training_iteration"]
        ]
        acc_kde_z_best = dataset_kde_z_best.properties["test_acc"]
    except:
        hue_kde_z_best = []
        epochs_kde_z_best = []
        acc_kde_z_best = []

    # Gan
    try:
        hue_gan = [
            # r"$S_{D30}$: sampled in top 30% z-distribution"
            r"$S_{G}$"
            for _ in range(len(dataset_gan.properties["test_acc"]))
        ]
        epochs_gan = [
            edx + epoch_offset for edx in dataset_gan.properties["training_iteration"]
        ]
        acc_gan = dataset_gan.properties["test_acc"]
    except:
        hue_gan = []
        epochs_gan = []
        acc_gan = []
    # Gan_best
    try:
        hue_gan_best = [
            r"$S_{G30}$" for _ in range(len(dataset_gan_best.properties["test_acc"]))
        ]
        epochs_gan_best = [
            edx + epoch_offset
            for edx in dataset_gan_best.properties["training_iteration"]
        ]
        acc_gan_best = dataset_gan_best.properties["test_acc"]
    except:
        hue_gan_best = []
        epochs_gan_best = []
        acc_gan_best = []

    ### labels
    # sample_size
    epochs_base = torch.unique(
        torch.tensor(dataset_baseline.properties["training_iteration"])
    ).shape[0]
    epochs_transfer = torch.unique(
        torch.tensor(dataset_direct.properties["training_iteration"])
    ).shape[0]
    n_original = len(dataset_baseline.properties["test_acc"]) // epochs_base
    n_direct = len(dataset_direct.properties["test_acc"]) // epochs_transfer
    n_uniform = len(dataset_uniform.properties["test_acc"]) // epochs_transfer
    n_train = len(dataset_train.properties["test_acc"]) // epochs_transfer
    n_best = len(dataset_best.properties["test_acc"]) // epochs_transfer
    n_kde_z_train = len(dataset_kde_z_train.properties["test_acc"]) // epochs_transfer
    n_kde_z_best = len(dataset_kde_z_best.properties["test_acc"]) // epochs_transfer
    n_gan = len(dataset_gan.properties["test_acc"]) // epochs_transfer
    n_gan_best = len(dataset_gan_best.properties["test_acc"]) // epochs_transfer

    ### prepare data
    x = []
    x.extend(epochs_baseline)
    x.extend(epochs_direct)
    x.extend(epochs_uniform)
    x.extend(epochs_train)
    x.extend(epochs_best)
    x.extend(epochs_kde_z_train)
    x.extend(epochs_kde_z_best)
    x.extend(epochs_gan)
    x.extend(epochs_gan_best)

    y = []
    y.extend(acc_baseline)
    y.extend(acc_direct)
    y.extend(acc_uniform)
    y.extend(acc_train)
    y.extend(acc_best)
    y.extend(acc_kde_z_train)
    y.extend(acc_kde_z_best)
    y.extend(acc_gan)
    y.extend(acc_gan_best)

    c = []
    c.extend(hue_baseline)
    c.extend(hue_direct)
    c.extend(hue_uniform)
    c.extend(hue_train)
    c.extend(hue_best)
    c.extend(hue_kde_z_train)
    c.extend(hue_kde_z_best)
    c.extend(hue_gan)
    c.extend(hue_gan_best)

    ## reduce by plotting domain
    if len(plot_domains) == 0:
        x_plot = x
        y_plot = y
        c_plot = c
        labels = []

    else:
        domain_keys = []
        labels = []
        labels2 = []
        if "baseline" in plot_domains:
            if len(hue_baseline) > 0:
                domain_keys.append(hue_baseline[0])
                labels.append(
                    r"$B_{T}$: trained on " + f"{target} - {n_original} samples"
                )
                labels2.append(r"$B_{T}$")
        if "direct" in plot_domains:
            if len(hue_direct) > 0:
                domain_keys.append(hue_direct[0])
                labels.append(
                    r"$B_{F}$: pretrained on "
                    + f"{source} and finetuned on {target} - {n_direct} samples"
                )
                labels2.append(r"$B_{F}$")
        if "uniform" in plot_domains:
            if len(hue_uniform) > 0:
                domain_keys.append(hue_uniform[0])
                labels.append(
                    r"$S_{0}$: sampled uniform in latent" + f" - {n_uniform} samples"
                )
                labels2.append(r"$S_{U}$")
        if "train" in plot_domains:
            if len(hue_train) > 0:
                domain_keys.append(hue_train[0])
                labels.append(
                    r"$S_{N}$: neigborhood-based sampling" + f" - {n_train} samples"
                )
                labels2.append(r"$S_{Neigh}$")
        if "best" in plot_domains:
            if len(hue_best) > 0:
                domain_keys.append(hue_best[0])
                labels.append(
                    r"$S_{N30}$: top 30% neighborhood-based sampling"
                    + f" - {n_best} samples"
                )
                labels2.append(r"$S_{Neigh30}$")
        if "kde_z_train" in plot_domains:
            if len(hue_kde_z_train) > 0:
                domain_keys.append(hue_kde_z_train[0])
                labels.append(
                    r"$S_D$: sampled in z-distribution" + f" - {n_kde_z_train} samples"
                )
                labels2.append(r"$S_{KDE}$")
        if "kde_z_best" in plot_domains:
            if len(hue_kde_z_best) > 0:
                domain_keys.append(hue_kde_z_best[0])
                labels.append(
                    r"$S_{D30}$: sampled in top 30% z-distribution"
                    + f" - {n_kde_z_best} samples"
                )
                labels2.append(r"$S_{KDE30}$")
        if "gan" in plot_domains:
            if len(hue_gan) > 0:
                domain_keys.append(hue_gan[0])
                labels.append(
                    r"$S_{G}$: GAN generated embeddings" + f" - {n_gan} samples"
                )
                labels2.append(r"$S_{GAN}$")
        if "gan_best" in plot_domains:
            if len(hue_gan_best) > 0:
                domain_keys.append(hue_gan_best[0])
                labels.append(
                    r"$S_{G30}$: GAN generated embeddings" + f" - {n_gan_best} samples"
                )
                labels2.append(r"$S_{GAN30}$")
        print(f"plot domains: {domain_keys}")
        idx_plot = [idx for idx, cdx in enumerate(c) if cdx in domain_keys]

        x_plot = [x[idx] for idx in idx_plot]
        y_plot = [y[idx] for idx in idx_plot]
        c_plot = [c[idx] for idx in idx_plot]

    # filter first source domain epochs for source==target
    if source == target:
        plot_epochs = [
            0,
            1,
            5,
            10,
            15,
            20,
            25,
            26,
            27,
            28,
            30,
            35,
            40,
            45,
            50,
        ]
        # plot_epochs.extend(list(range(26, 51)))
        idx_plot = [idx for idx, edx in enumerate(x_plot) if edx in plot_epochs]
        x_plot2 = [x_plot[idx] for idx in idx_plot]
        y_plot2 = [y_plot[idx] for idx in idx_plot]
        c_plot2 = [c_plot[idx] for idx in idx_plot]
    else:
        plot_epochs = [
            0,
            1,
            2,
            3,
            5,
            10,
            15,
            20,
            25,
            30,
            35,
            40,
            45,
            50,
        ]
        # plot_epochs.extend(list(range(26, 51)))
        idx_plot = [idx for idx, edx in enumerate(x_plot) if edx in plot_epochs]
        x_plot2 = [x_plot[idx] for idx in idx_plot]
        y_plot2 = [y_plot[idx] for idx in idx_plot]
        c_plot2 = [c_plot[idx] for idx in idx_plot]

    #### palette
    palette = sns.color_palette()
    color_palette = {
        r"$B_{F}$": palette[0],  # blue
        r"$B_{T}$": palette[2],  # green
        r"$\hat{B}_{F}$": palette[9],  # cyan
        r"$S_{N30}$": palette[4],  # purple
        #             r"$S_{N}$":           palette[6], #pink
        r"$S_{N}$": palette[8],  # yellow
        r"$S_{D30}$": palette[3],  # red
        r"$S_{D}$": palette[1],  # orange
        r"$S_{U}$": "tab:blue",
        r"$S_{C}$": "tab:cyan",
        r"$S_{G}$": palette[7],  # grey
        r"$S_{G30}$": palette[7],  # grey
    }

    ### plots
    fig_path = population_path.joinpath("figures")
    fig_path.mkdir(exist_ok=True)

    ##### boxplot all epochs
    sns.set_style("whitegrid")

    fig, ax = plt.subplots(figsize=(25, 8), dpi=300)

    # ax 0
    sns.boxplot(
        x=x_plot, y=y_plot, hue=c_plot, fliersize=0.0, palette=color_palette, ax=ax
    )
    ax.set_xlabel("Epochs", fontsize=20)
    ax.set_ylabel("Accuracy", fontsize=20)
    ax.xaxis.set_tick_params(labelsize=18)
    ax.yaxis.set_tick_params(labelsize=18)
    legend = ax.get_legend()
    handles = legend.legendHandles
    legend.remove()
    ax.legend(handles=handles, labels=labels2, loc="best", fontsize=18)
    # sns.despine(offset=20, trim=True)
    fig.suptitle(f"{source} to {target}", y=0.96, fontsize=20)

    fig.tight_layout()
    Path("./figures").mkdir(exist_ok=True)
    fname = fig_path.joinpath(f"{source}_to_{target}_boxplot.png")
    fig.savefig(fname)
    fname = fig_path.joinpath(f"{source}_to_{target}_boxplot.pdf")
    fig.savefig(fname)

    ##### boxplot first epochs wrapped epochs
    sns.set_style("whitegrid")

    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

    # ax 0
    sns.boxplot(
        x=x_plot2, y=y_plot2, hue=c_plot2, fliersize=0.0, palette=color_palette, ax=ax
    )
    ax.set_xlabel("Epochs", fontsize=14)
    ax.set_ylabel("Accuracy", fontsize=14)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    legend = ax.get_legend()
    handles = legend.legendHandles
    legend.remove()
    # ax.legend(handles=handles, labels=labels, loc="best", fontsize=16)
    ax.legend(handles=handles, labels=labels2, loc="lower right", fontsize=14)
    sns.despine(offset=20, trim=True)
    fig.suptitle(f"{source} to {target}", y=0.96, fontsize=16)

    fig.tight_layout()
    Path("./figures").mkdir(exist_ok=True)
    fname = fig_path.joinpath(f"{source}_to_{target}_boxplot_wrapped.png")
    fig.savefig(fname)
    fname = fig_path.joinpath(f"{source}_to_{target}_boxplot_wrapped.pdf")
    fig.savefig(fname)

    ##### boxplot second half
    if source == target:
        fig, ax = plt.subplots(figsize=(25, 8), dpi=300)
        sns.set_style("whitegrid")
        sns.boxplot(
            x=x_plot, y=y_plot, hue=c_plot, fliersize=0.0, palette=color_palette, ax=ax
        )
        ax.set_xlabel("Epochs", fontsize=14)
        ax.set_ylabel("Accuracy", fontsize=14)
        ax.xaxis.set_tick_params(labelsize=12)
        ax.yaxis.set_tick_params(labelsize=12)
        legend = ax.get_legend()
        handles = legend.legendHandles
        legend.remove()
        ax.legend(handles=handles, labels=labels2, loc="best", fontsize=16)

        ax.set_xlim([id_offset - 0.5, 50.5])
        # ax.set_xticklabels(list(range(id_offset, 51)))

        sns.despine(offset=20, trim=True)
        fig.suptitle(f"{source} to {target}", y=0.96, fontsize=16)

        # fig.tight_layout()
        fname = fig_path.joinpath(f"{source}_to_{target}_boxplot_detail.png")
        fig.savefig(fname)
        fname = fig_path.joinpath(f"{source}_to_{target}_boxplot_detail.pdf")
        fig.savefig(fname)

    ####### distplot shared axis
    # get interesting epochs
    if source == target:
        epoch_choice_list = [
            id_offset,
            id_offset + 3,
            id_offset + 10,
            epochs_base - 1,
        ]
    else:
        epoch_choice_list = [0, 5, 10, epochs_base - 1]

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(25, 24), nrows=4, sharex=True, dpi=300)

    # slice:
    epoch_choice = epoch_choice_list[0]
    idx_epoch = [idx for idx, edx in enumerate(x_plot) if edx == epoch_choice]
    y_slice = [y_plot[idx] for idx in idx_epoch]
    c_slice = [c_plot[idx] for idx in idx_epoch]
    sns.histplot(
        x=y_slice,
        hue=c_slice,
        stat="probability",
        ax=ax[0],
        kde=True,
        legend=True,
        binwidth=0.002,
        common_norm=False,
    )
    ax[0].set_title(f"test accuracy distribution - epoch {epoch_choice}", fontsize=16)
    legend = ax[0].get_legend()
    handles = legend.legendHandles
    legend.remove()
    ax[0].legend(handles=handles, labels=labels, loc="upper center", fontsize=16)
    ax[0].set_xlabel("test accuracy", fontsize=12)

    # slice:
    epoch_choice = epoch_choice_list[1]
    idx_epoch = [idx for idx, edx in enumerate(x_plot) if edx == epoch_choice]
    y_slice = [y_plot[idx] for idx in idx_epoch]
    c_slice = [c_plot[idx] for idx in idx_epoch]
    sns.histplot(
        x=y_slice,
        hue=c_slice,
        ax=ax[1],
        stat="probability",
        kde=True,
        binwidth=0.002,
        common_norm=False,
    )
    legend = ax[1].get_legend()
    handles = legend.legendHandles
    legend.remove()
    ax[1].legend(handles=handles, labels=labels, loc="upper center", fontsize=16)
    ax[1].set_title(f"test accuracy distribution - epoch {epoch_choice}", fontsize=16)
    ax[1].set_xlabel("test accuracy", fontsize=12)

    # slice:
    epoch_choice = epoch_choice_list[2]
    idx_epoch = [idx for idx, edx in enumerate(x_plot) if edx == epoch_choice]
    y_slice = [y_plot[idx] for idx in idx_epoch]
    c_slice = [c_plot[idx] for idx in idx_epoch]
    sns.histplot(
        x=y_slice,
        hue=c_slice,
        ax=ax[2],
        stat="probability",
        kde=True,
        binwidth=0.002,
        common_norm=False,
    )
    legend = ax[2].get_legend()
    handles = legend.legendHandles
    legend.remove()
    ax[2].legend(handles=handles, labels=labels, loc="upper center", fontsize=16)
    ax[2].set_title(f"test accuracy distribution - epoch {epoch_choice}", fontsize=16)
    ax[2].set_xlabel("test accuracy", fontsize=12)

    # slice:
    epoch_choice = epoch_choice_list[3]
    idx_epoch = [idx for idx, edx in enumerate(x_plot) if edx == epoch_choice]
    y_slice = [y_plot[idx] for idx in idx_epoch]
    c_slice = [c_plot[idx] for idx in idx_epoch]
    sns.histplot(
        x=y_slice,
        hue=c_slice,
        ax=ax[3],
        stat="probability",
        kde=True,
        binwidth=0.002,
        common_norm=False,
    )
    legend = ax[3].get_legend()
    handles = legend.legendHandles
    legend.remove()
    ax[3].legend(handles=handles, labels=labels, loc="upper center", fontsize=16)
    ax[3].set_title(f"test accuracy distribution - epoch {epoch_choice}", fontsize=16)
    ax[3].set_xlabel("test accuracy", fontsize=12)

    fig.suptitle(f"{source} to {target} Transfer Learning", y=0.92, fontsize=18)

    fname = fig_path.joinpath(f"{source}_to_{target}_histogram.png")
    fig.savefig(fname)
    fname = fig_path.joinpath(f"{source}_to_{target}_histogram.pdf")
    fig.savefig(fname)

    ####### distplot individual axis
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(25, 24), nrows=4, sharex=False, dpi=300)

    # slice:
    epoch_choice = epoch_choice = epoch_choice_list[0]
    idx_epoch = [idx for idx, edx in enumerate(x_plot) if edx == epoch_choice]
    y_slice = [y_plot[idx] for idx in idx_epoch]
    c_slice = [c_plot[idx] for idx in idx_epoch]
    sns.histplot(
        x=y_slice,
        hue=c_slice,
        stat="probability",
        ax=ax[0],
        kde=True,
        legend=True,
        binwidth=0.002,
        common_norm=False,
    )
    ax[0].set_title(f"test accuracy distribution - epoch {epoch_choice}", fontsize=16)
    legend = ax[0].get_legend()
    handles = legend.legendHandles
    legend.remove()
    ax[0].legend(handles=handles, labels=labels, loc="upper center", fontsize=16)
    ax[0].set_xlabel("test accuracy", fontsize=12)

    # slice:
    epoch_choice = epoch_choice_list[1]
    idx_epoch = [idx for idx, edx in enumerate(x_plot) if edx == epoch_choice]
    y_slice = [y_plot[idx] for idx in idx_epoch]
    c_slice = [c_plot[idx] for idx in idx_epoch]
    sns.histplot(
        x=y_slice,
        hue=c_slice,
        ax=ax[1],
        stat="probability",
        kde=True,
        binwidth=0.002,
        common_norm=False,
    )
    legend = ax[1].get_legend()
    handles = legend.legendHandles
    legend.remove()
    ax[1].legend(handles=handles, labels=labels, loc="upper center", fontsize=16)
    ax[1].set_title(f"test accuracy distribution - epoch {epoch_choice}", fontsize=16)
    ax[1].set_xlabel("test accuracy", fontsize=12)

    # slice:
    epoch_choice = epoch_choice_list[2]
    idx_epoch = [idx for idx, edx in enumerate(x_plot) if edx == epoch_choice]
    y_slice = [y_plot[idx] for idx in idx_epoch]
    c_slice = [c_plot[idx] for idx in idx_epoch]
    sns.histplot(
        x=y_slice,
        hue=c_slice,
        ax=ax[2],
        stat="probability",
        kde=True,
        binwidth=0.002,
        common_norm=False,
    )
    legend = ax[2].get_legend()
    handles = legend.legendHandles
    legend.remove()
    ax[2].legend(handles=handles, labels=labels, loc="upper center", fontsize=16)
    ax[2].set_title(f"test accuracy distribution - epoch {epoch_choice}", fontsize=16)
    ax[2].set_xlabel("test accuracy", fontsize=12)

    # slice:
    epoch_choice = epoch_choice_list[3]
    idx_epoch = [idx for idx, edx in enumerate(x_plot) if edx == epoch_choice]
    y_slice = [y_plot[idx] for idx in idx_epoch]
    c_slice = [c_plot[idx] for idx in idx_epoch]
    sns.histplot(
        x=y_slice,
        hue=c_slice,
        ax=ax[3],
        stat="probability",
        kde=True,
        binwidth=0.002,
        common_norm=False,
    )
    legend = ax[3].get_legend()
    handles = legend.legendHandles
    legend.remove()
    ax[3].legend(handles=handles, labels=labels, loc="upper center", fontsize=16)
    ax[3].set_title(f"test accuracy distribution - epoch {epoch_choice}", fontsize=16)
    ax[3].set_xlabel("test accuracy", fontsize=12)

    fig.suptitle(f"{source} to {target} Transfer Learning", y=0.92, fontsize=18)

    fname = fig_path.joinpath(f"{source}_to_{target}_histogram_detail.png")
    fig.savefig(fname)
    fname = fig_path.joinpath(f"{source}_to_{target}_histogram_detail.pdf")
    fig.savefig(fname)

    ## compute statistics
    for epochdx in epoch_choice_list:
        print(f"compute statistics of significance for epoch {epochdx}")
        # slice for epoch
        idx_epoch = [idx for idx, edx in enumerate(x) if edx == epochdx]
        y_slice = [y[idx] for idx in idx_epoch]
        c_slice = [c[idx] for idx in idx_epoch]
        print(f"y_slice: {len(y_slice)} samples")
        print(f"c_slice: {len(c_slice)} samples")

        acc_dict = {}
        # slice for hue
        if hue_baseline is not None:
            try:
                acc_base = [
                    ydx for ydx, cdx in zip(y_slice, c_slice) if cdx == hue_baseline[0]
                ]
                print(f"acc_base: {len(acc_base)} samples")
                if len(acc_base) > 0:
                    acc_dict["base"] = acc_base
            except:
                print("error loading basline data for statistics")
        if hue_direct is not None:
            try:
                acc_direct = [
                    ydx for ydx, cdx in zip(y_slice, c_slice) if cdx == hue_direct[0]
                ]
                print(f"acc_direct: {len(acc_direct)} samples")
                if len(acc_direct) > 0:
                    acc_dict["direct"] = acc_direct
            except:
                print("error loading direct data for statistics")

        if hue_uniform is not None:
            try:
                acc_uniform = [
                    ydx for ydx, cdx in zip(y_slice, c_slice) if cdx == hue_uniform[0]
                ]
                print(f"acc_uniform: {len(acc_uniform)} samples")
                if len(acc_uniform) > 0:
                    acc_dict["uniform"] = acc_uniform
            except:
                print("error loading uniform data for statistics")
        if hue_train is not None:
            try:
                acc_train = [
                    ydx for ydx, cdx in zip(y_slice, c_slice) if cdx == hue_train[0]
                ]
                print(f"acc_train: {len(acc_train)} samples")
                if len(acc_train) > 0:
                    acc_dict["train"] = acc_train
            except:
                print("error loading umap train data for statistics")
        if hue_best is not None:
            try:
                acc_best = [
                    ydx for ydx, cdx in zip(y_slice, c_slice) if cdx == hue_best[0]
                ]
                print(f"acc_best: {len(acc_best)} samples")
                if len(acc_best) > 0:
                    acc_dict["best"] = acc_best
            except:
                print("error loading umap best data for statistics")
        if hue_kde_z_train is not None:
            try:
                acc_kde_z_train = [
                    ydx
                    for ydx, cdx in zip(y_slice, c_slice)
                    if cdx == hue_kde_z_train[0]
                ]
                print(f"acc_train: {len(acc_kde_z_train)} samples")
                if len(acc_kde_z_train) > 0:
                    acc_dict["kde_z_train"] = acc_kde_z_train
            except:
                print("error loading kde_train data for statistics")
        if hue_kde_z_best is not None:
            try:
                acc_kde_z_best = [
                    ydx
                    for ydx, cdx in zip(y_slice, c_slice)
                    if cdx == hue_kde_z_best[0]
                ]
                print(f"acc_best: {len(acc_kde_z_best)} samples")
                if len(acc_kde_z_best) > 0:
                    acc_dict["kde_z_best"] = acc_kde_z_best
            except:
                print("error loading kde_best data for statistics")
        if hue_gan is not None:
            try:
                acc_gan = [
                    ydx for ydx, cdx in zip(y_slice, c_slice) if cdx == hue_gan[0]
                ]
                print(f"acc_best: {len(acc_gan)} samples")
                if len(acc_gan) > 0:
                    acc_dict["gan"] = acc_gan
            except:
                print("error loading gan data for statistics")
        if hue_gan_best is not None:
            try:
                acc_gan_best = [
                    ydx for ydx, cdx in zip(y_slice, c_slice) if cdx == hue_gan_best[0]
                ]
                print(f"acc_best: {len(acc_gan_best)} samples")
                if len(acc_gan_best) > 0:
                    acc_dict["gan_best"] = acc_gan_best
            except:
                print("error loading gan_best data for statistics")
        stats_path = population_path.joinpath("statistics")
        stats_path.mkdir(exist_ok=True)
        compute_statistics(
            acc_dict=acc_dict,
            path=stats_path,
            print_dfs=True,
            prefix=f"epoch_{epochdx}",
        )


#########################################################################
### add CI computation ##################################################
#########################################################################


def compute_mwu_ci(
    df_wmu, x, y, alternative="two-sided", confidence: float = 0.95
) -> list:
    """
    # Calculating confidence intervals for some non-parametric analyses
    # MICHAEL J CAMPBELL, MARTIN J GARDNER
    # BRITISH MEDICAL JOURNAL VOLUME 296 21 MAY 1988
    # https://stackoverflow.com/questions/51845606/python-mann-whitney-confidence-interval

    """

    x = np.asarray(x)
    y = np.asarray(y)
    # Confidence interval for the (difference in) medians
    # Campbell and Gardner 2000
    if alternative == "two-sided":
        alpha = 1.0 - confidence
        conf = 1.0 - alpha / 2  # 0.975
    else:
        conf = confidence
    N = st.norm.ppf(conf)
    ct1, ct2 = len(x), len(y)
    diffs = sorted([i - j for i in x for j in y])
    k = int(round(ct1 * ct2 / 2 - (N * (ct1 * ct2 * (ct1 + ct2 + 1) / 12) ** 0.5)))
    ci = [diffs[k], diffs[len(diffs) - k]]
    if alternative == "greater":
        ci[1] = np.inf
    elif alternative == "less":
        ci[0] = -np.inf
    # Rename CI
    ci_name = "CI%.0f%%" % (100 * confidence)
    df_wmu = df_wmu.assign(ci_name=[ci])
    return df_wmu


#########################################################################
### add CI computation ##################################################
#########################################################################


def compute_statistics(
    acc_dict: dict,
    path: Path,
    print_dfs: bool = False,
    confidence: float = 0.95,
    prefix: str = "",
):
    """
    Computes the pairwise statistics (significance, effect strength, CI)
    for the accuracies of populations in acc_dict
    and saves them in path
    """
    # init outputs
    df_normality = pd.DataFrame()
    df_ttest = pd.DataFrame()
    df_wmu = pd.DataFrame()
    population_lst = list(acc_dict.keys())
    dict_stats = {
        "pop": [],
        "mean": [],
        "std": [],
        "median": [],
        f"median_ci_{confidence:.2f}": [],
        f"mean_ci_{confidence:.2f}": [],
        "quantiles": [],
    }
    # iterate over populations
    for pop1_key in tqdm.tqdm(population_lst):
        # get first pop
        pop1 = acc_dict[pop1_key]
        # compute basic stats
        dict_stats["pop"].append(pop1_key)
        dict_stats["mean"].append(np.mean(pop1))
        dict_stats["std"].append(np.std(pop1))
        dict_stats["median"].append(np.median(pop1))
        qs = [0, 0.05, 0.3, 0.5, 0.7, 0.95, 1.0]
        dict_stats["quantiles"].append(np.quantile(a=pop1, q=qs))
        res = st.bootstrap(
            (pop1,), np.median, confidence_level=confidence, random_state=42
        )
        median_ci = [res.confidence_interval.low, res.confidence_interval.high]
        dict_stats[f"median_ci_{confidence:.2f}"].append(median_ci)
        res = st.bootstrap(
            (pop1,), np.mean, confidence_level=confidence, random_state=42
        )
        mean_ci = [res.confidence_interval.low, res.confidence_interval.high]
        dict_stats[f"mean_ci_{confidence:.2f}"].append(mean_ci)
        # normality
        df_normality_tmp = pg.normality(pop1)  # Univariate normality
        df_normality_tmp = df_normality_tmp.assign(pop=[pop1_key])
        df_normality = df_normality.append(df_normality_tmp)
        for pop2_key in population_lst:
            # get second pop
            pop2 = acc_dict[pop2_key]
            # compute ttest
            df_ttest_tmp = pg.ttest(
                pop1, pop2, paired=False, alternative="two-sided", confidence=confidence
            )
            # adjust dataframe
            df_ttest_tmp = df_ttest_tmp.assign(pop1=[pop1_key], pop2=[pop2_key])
            # add current dframe to df_ttest
            df_ttest = df_ttest.append(df_ttest_tmp)
            # compute wmu
            try:
                df_wmu_tmp = pg.mwu(
                    pop1, pop2, alternative="two-sided", confidence=confidence
                )
            except TypeError:
                # confidence not yet implemented
                df_wmu_tmp = pg.mwu(pop1, pop2, alternative="two-sided")
                df_wmu_tmp = compute_mwu_ci(
                    df_wmu=df_wmu_tmp,
                    x=pop1,
                    y=pop2,
                    alternative="two-sided",
                    confidence=confidence,
                )
            # adjust dataframe
            df_wmu_tmp = df_wmu_tmp.assign(pop1=[pop1_key], pop2=[pop2_key])
            # add current dframe to df_wmu
            df_wmu = df_wmu.append(df_wmu_tmp)
    df_stats = pd.DataFrame(dict_stats)
    fname = path.joinpath(prefix + "pop_stats.json")
    df_stats.to_json(fname)
    # reset index
    df_normality = df_normality.reset_index()
    df_ttest = df_ttest.reset_index()
    df_wmu = df_wmu.reset_index()
    # save
    fname = path.joinpath(prefix + "normality.json")
    df_normality.to_json(fname)
    # save
    fname = path.joinpath(prefix + "ttest.json")
    df_ttest.to_json(fname)
    # save
    fname = path.joinpath(prefix + "wmu.json")
    df_wmu.to_json(fname)
    # print
    if print_dfs:
        print("normality")
        print(df_normality)
        #
        print("t-test")
        print(df_ttest)
        #
        print("Mann Whitney U Test")
        print(df_wmu)
