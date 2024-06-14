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

import copy

from SANE.sampling.condition_bn import condition_checkpoints, check_equivalence
from SANE.sampling.load_dataset import load_datasets_from_config
from SANE.sampling.evaluate_ensemble import evaluate_ensemble
from SANE.sampling.evaluate_single_model import evaluate_single_model
from SANE.sampling.de_normalize import de_normalize_checkpoint
from SANE.sampling.get_anchor_embeddings import (
    get_random_anchor_embeddings,
    get_anchor_embeddings,
)
from SANE.sampling.sample_models import sample_models

from pathlib import Path


##
def sample_model_evaluation_bootstrapped(
    ae_model,
    sample_config: dict,
    finetuning_epochs: int,
    repetitions: int,
    anchor_ds_path: str,
    reference_dataset_path: str,
    bootstrap_iterations: int,
    bootstrap_samples: int,
    bootstrap_keep_top_n: int,
    mode: str = "individual",  # 'individual','token,'joint'
    norm_mode: Optional[str] = None,
    layer_norms: Optional[dict] = None,
    batch_size: int = 0,
    reset_classifier: bool = False,
    halo: bool = False,
    halo_wse: int = 156,
    halo_hs: int = 64,
    bn_condition_iters: int = 0,
    mu_glob: float = 0.0,
    sigma_glob: float = 10.0,
    ensemble: bool = False,
    anchor_sample_number: int = 0,
    drop_samples_to_path: Optional[str | Path] = None,
) -> dict:
    """
    runs evaluation pipeline.
    samples hyper-representation model to generate checkpoints, finetunes checkpoints on downstream task and evaluates finetuned checkpoints
    Args:
        ae_model (hyper-representation): hyper-representation model
        sample_config (dict): dictionary containing config for sampled model
        finetuning_epochs (int): number of epochs to finetune
        repetitions (int): number of repetitions to finetune and evaluate
        anchor_ds_path: path to model dataset, which are used to fit the distributions to sample from
        reference_dataset_path: path to image dataset, which are used to evaluate the sampled models by for bootstrapping
        bootstrap_iterations (int): number of bootstrap iterations
        bootstrap_samples (int): number of samples per bootstrap iteration
        bootstrap_keep_top_n (int): number of top performing models to keep for next bootstrap iteration
        mode (str): sampling mode, either 'individual','token,'joint'
        norm_mode (str): normalization mode, either 'standardize' or 'normalize'
        layer_norms (dict): dictionary containing normalization parameters
        batch_size (int): batch size for finetuning
        reset_classifier (bool): whether to reset classifier weights after sampling
        halo (bool, optional): use halo-windows for encoding / decoding, instead of passing the entire sequence in one go. Defaults to False.
        halo_wse (int, optional): size of haloed-window. Defaults to 156.
        halo_hs (int, optional): size of the halo around the window. Defaults to 64.
        bn_condition_iters: (int, optional): if nonzero, perform conditioning iterations on train/val image dataset to tune bn statistics (only stats, no weight udpates)
        mu_glob: (float, optional): mean of normal distribution to draw initial samples from
        sigma_glob: (float, optional): standard deviation of normal distribution to draw initial samples from
        ensemble (bool, optional): whether to evaluate model ensemble
        anchor_sample_number (int, optional): number of anchor samples to draw from anchor dataset. if 0, use all samples
    Returns:
        dict: dictionary containing evaluation results
    """
    assert (
        bootstrap_samples >= repetitions
    ), f"bootstrap number {bootstrap_samples} needs to be larger than repetitions {repetitions}"
    # init output
    results = {}
    # get reference model tokens
    logging.info("sampling:: create reference checkoint")
    sample_config["scheduler::steps_per_epoch"] = 123
    sample_config["device"] = "cpu"  # to have sampled state_dicts on cpu
    module = NNmodule(sample_config, cuda=False)
    checkpoint_ref = module.model.state_dict()

    # get first anchor embeddings
    logging.info(
        f"sampling:: get first anchor embeddings from anchor_ds_path: {anchor_ds_path}"
    )
    if anchor_ds_path is not None:
        anchor_z, anchor_pos, anchor_w_shape = get_anchor_embeddings(
            anchor_ds_path=anchor_ds_path,
            ae_model=ae_model,
            batch_size=batch_size,
            halo=halo,
            halo_wse=halo_wse,
            halo_hs=halo_hs,
            samples=anchor_sample_number,
        )
    else:
        logging.info(
            f"no anchor ds provided: draw anchor samples from normal distribution"
        )
        tokensize = ae_model.config["ae:i_dim"]
        lat_dim = ae_model.config["ae:lat_dim"]
        anchor_z, anchor_pos, anchor_w_shape = get_random_anchor_embeddings(
            n_samples=anchor_sample_number,
            tokensize=tokensize,
            lat_dim=lat_dim,
            sample_config=sample_config,
            mu_glob=mu_glob,
            sigma_glob=sigma_glob,
        )

    # iterate over bootstrap iterations
    logging.info(f"start bootstrap iterations")
    for bdx in range(bootstrap_iterations):
        # sample models
        logging.info(f"bootstrap iteration {bdx} sampling:: sample models")
        (checkpoints, anchor_z, anchor_pos, anchor_w_shape) = sample_models(
            ae_model=ae_model,
            checkpoint_ref=checkpoint_ref,
            anchor_z=anchor_z,
            anchor_pos=anchor_pos,
            anchor_w_shape=anchor_w_shape,
            mode=mode,
            repetitions=bootstrap_samples,
            batch_size=batch_size,
            return_new_anchor=True,
            reset_classifier=reset_classifier,
            halo=halo,
            halo_wse=halo_wse,
            halo_hs=halo_hs,
        )

        # check if checkpoints are the same
        if check_equivalence(checkpoints[0], checkpoints[-1]):
            logging.warning(f"monitoring: same checkpoints after sampling")

        # de-normalize checkoints
        logging.info("sampling:: de-normalize checkpoints")
        if norm_mode is not None:
            for idx in range(len(checkpoints)):
                checkpoints[idx] = de_normalize_checkpoint(
                    checkpoints[idx], layers=layer_norms, mode=norm_mode
                )
        if check_equivalence(checkpoints[0], checkpoints[-1]):
            logging.warning(f"monitoring: same checkpoints after normalization")

        # condition bn of checkpoints
        if bn_condition_iters > 0:
            logging.info(f"sampling: condition bn layers")
            checkpoints = condition_checkpoints(
                checkpoints, sample_config, bn_condition_iters
            )
        if check_equivalence(checkpoints[0], checkpoints[-1]):
            logging.warning(f"monitoring: same checkpoints after conditioning")

        # bootstrap: evaluate models on reference_dataset, keep  only the best ones
        logging.info("sampling:: bootstrap")
        config_bootstrap = copy.deepcopy(sample_config)
        # checkpoints = bootstrap_model_evaluation(
        model_index_sorted, model_perf_sorted = bootstrap_model_evaluation(
            checkpoints,
            config_bootstrap,
            reference_dataset_path=reference_dataset_path,
        )

        # keep top n checkpoints and repeat bootstrap
        model_index_sorted = model_index_sorted[:bootstrap_keep_top_n]
        anchor_z = anchor_z[model_index_sorted]
        model_perf_best = model_perf_sorted[:bootstrap_keep_top_n]
        model_perf_disc = model_perf_sorted[bootstrap_keep_top_n:]

        # log performance of boostrapped models, to see if bootstrapping helps
        results[f"bootstrap_it_{bdx}_model_perf_best_mean"] = np.mean(model_perf_best)
        results[f"bootstrap_it_{bdx}_model_perf_best_std"] = np.std(model_perf_best)
        results[f"bootstrap_it_{bdx}_model_perf_disc_mean"] = np.mean(model_perf_disc)
        results[f"bootstrap_it_{bdx}_model_perf_disc_std"] = np.std(model_perf_disc)

        logging.info(
            f"bootstrap iteration {bdx} - best model val acc {np.mean(model_perf_best)}+-{np.std(model_perf_best)}"
        )
        logging.info(
            f"bootstrap iteration {bdx} - other model val acc {np.mean(model_perf_disc)}+-{np.std(model_perf_disc)}"
        )
    # cleanup
    del ae_model, checkpoint_ref, anchor_z, anchor_pos, anchor_w_shape

    logging.info("finished bootstrapping. evaluate best models")
    # keep only repetitions checkpoints
    checkpoints = checkpoints[:repetitions]

    assert (
        len(checkpoints) == repetitions
    ), f"after bootstrapping, checkpoints {len(checkpoints)} need to match repetitions {repetitions}"

    if drop_samples_to_path is not None:
        logging.info(f"sampling: drop samples to path {drop_samples_to_path}")
        drop_samples_to_path.mkdir(exist_ok=True, parents=True)
        for idx in range(len(checkpoints)):
            checkpoint = checkpoints[idx]
            checkpoint_path = drop_samples_to_path.joinpath(f"checkpoint_{idx}.pt")
            torch.save(checkpoint, checkpoint_path)
        return results

    # evaluate models
    logging.info("sampling:: evaluate models")
    for rep in range(repetitions):
        # sample model
        checkpoint = checkpoints[rep]
        # finetune and evaluate model
        res_tmp = evaluate_single_model(sample_config, checkpoint, finetuning_epochs)
        # append results
        for k in res_tmp.keys():
            results[f"model_{rep}_{k}"] = res_tmp[k]

    # aggregate results over models
    for k in res_tmp.keys():
        res_tmp = []
        for rep in range(repetitions):
            res_tmp.append(results[f"model_{rep}_{k}"])

        results[f"{k}_mean"] = []
        results[f"{k}_std"] = []
        for idx in range(len(res_tmp[0])):
            res_ep = [res_tmp[jdx][idx] for jdx in range(len(res_tmp))]
            results[f"{k}_mean"].append(np.mean(res_ep))
            results[f"{k}_std"].append(np.std(res_ep))

    # compute model ensemble
    if ensemble:
        logging.info("sampling:: compute model ensemble")
        res_ens = evaluate_ensemble(sample_config, checkpoints, finetuning_epochs)
        for k in res_ens.keys():
            results[f"ensemble_{k}"] = res_ens[k]

    # return results
    return results


def bootstrap_model_evaluation(checkpoints, config, reference_dataset_path):
    """
    evaluates a list of checkpoints on a single task
    keeps only the top n checkpoints
    Args:
        checkpoints (list): list of state dicts of the model
        config (dict): dictionary containing config for the model
        reference_dataset_path: path to image dataset, which are used to evaluate the sampled models by for bootstrapping
    Returns:
        list: list of top checkpoints
    """
    # fix config with new reference dataset
    config["dataset::dump"] = reference_dataset_path
    # load dataset
    trainlaoder, _, valloader = load_datasets_from_config(config)
    # use validation set as default
    if valloader is not None:
        dataloader = valloader
    else:
        # fallback to trainset if no validation set is available
        dataloader = trainlaoder
    # set dummy parameters for one cycle scheduler  (if used)
    config["training::epochs_train"] = 1
    config["scheduler::steps_per_epoch"] = 123
    # init model
    logging.info("initialize sampled model")
    module = NNmodule(config, cuda=True, verbosity=0)
    #
    model_ind_perf = []
    #
    for idx in range(len(checkpoints)):
        # get checkpoint
        check = checkpoints[idx]
        # load checkpoint
        logging.info("load checkpoint model")
        module.model.load_state_dict(check)
        # validate
        loss_val, acc_val = module.test_epoch(dataloader, 0)
        # append list
        model_ind_perf.append((acc_val, idx))

    # sort model_ind_perf by acc_val in descending order
    model_ind_perf = sorted(model_ind_perf, key=lambda x: x[0], reverse=True)
    assert (
        model_ind_perf[0][0] >= model_ind_perf[-1][0]
    ), "model_ind_perf not sorted correctly"

    # # extract keep_top_n best models from model_ind_perf
    # model_ind_perf = model_ind_perf[:keep_top_n]
    # ectract indices of (acc,  ind) tuples
    model_ind = [ddx[1] for ddx in model_ind_perf]
    model_perf = [ddx[0] for ddx in model_ind_perf]
    return model_ind, model_perf
    # apply index on top_n  checkpoints
    # checkpoints_top_n = [checkpoints[idx] for idx in model_ind]
    # return checkpoints_top_n
