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
def sample_model_evaluation(
    ae_model,
    sample_config: dict,
    finetuning_epochs: int,
    repetitions: int,
    anchor_ds_path: str,
    mode: str = "individual",  # 'individual','token,'joint'
    norm_mode: Optional[str] = None,
    layer_norms: Optional[dict] = None,
    batch_size: int = 0,
    reset_classifier: bool = False,
    halo: bool = False,
    halo_wse: int = 156,
    halo_hs: int = 64,
    bn_condition_iters: int = 0,
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
        anchor_ds_path (str): path to anchor dataset
        mode (str, optional): sampling mode. Defaults to "individual".
        norm_mode (Optional[str], optional): normalization mode. Defaults to None.
        layer_norms (Optional[dict], optional): normalization parameters. Defaults to None.
        batch_size (int, optional): batch size for finetuning. Defaults to 0.
        reset_classifier (bool, optional): reset classifier before finetuning. Defaults to False.
        halo (bool, optional): use halo-windows for encoding / decoding, instead of passing the entire sequence in one go. Defaults to False.
        halo_wse (int, optional): size of haloed-window. Defaults to 156.
        halo_hs (int, optional): size of the halo around the window. Defaults to 64.
        bn_condition_iters: (int, optional): if nonzero, perform conditioning iterations on train/val image dataset to tune bn statistics (only stats, no weight udpates)
        anchor_sample_number (int, optional): number of anchor samples to draw from anchor dataset. if 0, use all samples
    Returns:
        dict: dictionary containing evaluation results
    """
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
    print(
        f"get anchor embeddings with halo: {halo} window size: {halo_wse} halo size: {halo_hs}"
    )
    anchor_z, anchor_pos, anchor_w_shape = get_anchor_embeddings(
        anchor_ds_path=anchor_ds_path,
        ae_model=ae_model,
        batch_size=batch_size,
        halo=halo,
        halo_wse=halo_wse,
        halo_hs=halo_hs,
        samples=anchor_sample_number,
    )

    # sample models
    logging.info("sampling:: sample models")
    checkpoints = sample_models(
        ae_model=ae_model,
        checkpoint_ref=checkpoint_ref,
        anchor_z=anchor_z,
        anchor_pos=anchor_pos,
        anchor_w_shape=anchor_w_shape,
        mode=mode,
        repetitions=repetitions,
        batch_size=batch_size,
        return_new_anchor=False,
        reset_classifier=reset_classifier,
        halo=halo,
        halo_wse=halo_wse,
        halo_hs=halo_hs,
    )

    # cleanup
    del ae_model, checkpoint_ref, anchor_z, anchor_pos, anchor_w_shape

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
