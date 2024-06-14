import torch
import torch.nn as nn
import numpy as np
import random
from SANE.models.def_net import NNmodule
import logging
import weightwatcher as ww
from SANE.datasets.dataset_auxiliaries import tokens_to_checkpoint


### just an identity model, maps input on itself
class IdentityModel(nn.Module):
    def __init__(self):
        super(IdentityModel, self).__init__()

    def forward(self, x, p):
        """
        just flatten the weight tokens
        this ignores the fact that there is a lot of padding involved here
        """
        return x.flatten(start_dim=1)

    def forward_embeddings(self, x, p):
        return x.flatten(start_dim=1)


class LayerQuintiles(nn.Module):
    def __init__(
        self,
    ):
        super(LayerQuintiles, self).__init__()

    def forward(self, x, p):
        z = self.compute_layer_quintiles(x, p)
        return z

    def forward_embeddings(self, x, p):
        z = self.compute_layer_quintiles(x, p)
        return z

    def compute_layer_quintiles(self, weights, pos):
        # weights need to be on the cpu for numpy
        weights = weights.to(torch.device("cpu"))
        quantiles = [1, 25, 50, 75, 99]
        features = []
        # get unique layers, assume pos.shape = [n_samples, n_tokens, 3]
        layers = torch.unique(pos[0, :, 1])
        # iterate over layers
        for idx, layer in enumerate(layers):
            # get slices
            index_kernel = torch.where(pos[0, :, 1] == layer)[0]
            index_kernel = index_kernel.to("cpu")
            # compute kernel stat values
            wtmp = weights[:, index_kernel].detach().flatten(start_dim=1).numpy()
            # print(wtmp.shape)
            # # compute kernel stat values
            features_ldx_weights = np.percentile(
                a=wtmp,
                q=quantiles,
                axis=1,
            )
            features_ldx_weights = torch.tensor(features_ldx_weights)
            # # transpose to [n_samples, n_features]
            features_ldx_weights = torch.einsum("ij->ji", features_ldx_weights)
            # print(features_ldx_weights.shape)
            mean_ldx_weights = torch.tensor(np.mean(a=wtmp, axis=1)).unsqueeze(dim=1)
            var_ldx_weights = torch.tensor(np.var(a=wtmp, axis=1)).unsqueeze(dim=1)
            # print(mean_ldx_weights.shape)
            # print(var_ldx_weights.shape)
            features.extend([mean_ldx_weights, var_ldx_weights, features_ldx_weights])
        # put together
        z = torch.cat(features, dim=1)
        return z


import random


### just an identity model, maps input on itself
class WeightChunkModel(nn.Module):
    def __init__(self, chunksize, pos: str = "start"):
        super(WeightChunkModel, self).__init__()
        self.chunksize = chunksize
        self.pos = pos

    def forward(self, x, p):
        """
        just flatten the weight tokens
        this ignores the fact that there is a lot of padding involved here
        """
        x = x.flatten(start_dim=1)
        # draw chunk out of second dimensino
        if self.pos == "start":
            return x[:, : self.chunksize]
        elif self.pos == "end":
            return x[:, -self.chunksize :]
        # else: random
        # get lenght of token sequence
        max_len = x.shape[-1]
        assert max_len >= self.chunksize
        # sample start
        if max_len == self.chunksize:
            idx_start = 0
        else:
            idx_start = random.randint(0, max_len - self.chunksize)
        idx_end = idx_start + self.chunksize
        # get index tensor
        idx = torch.arange(start=idx_start, end=idx_end, device=x.device)
        # apply window
        x = torch.index_select(x, -1, idx)
        return x

    def forward_embeddings(self, x, p):
        return self.forward(x, p)


### just an identity model, maps input on itself
class WeightWatcherModel(nn.Module):

    def __init__(
        self,
        reference_model_config,
        wwkey=[
            "log_norm",
            "alpha",
            "alpha_weighted",
            "log_alpha_norm",
            "log_spectral_norm",
            "stable_rank",
        ],
    ):
        super(WeightWatcherModel, self).__init__()
        self.reference_model_config = reference_model_config
        self.reference_model = NNmodule(reference_model_config, cuda=True, verbosity=0)
        self.reference_checkpoint = self.reference_model.model.state_dict()
        self.wwkey = wwkey

    def forward(self, x, p):
        """
        just flatten the weight tokens
        this ignores the fact that there is a lot of padding involved here
        """
        out = []
        for idx in range(x.shape[0]):
            # cast sample to checkpoint
            checkpoint = tokens_to_checkpoint(
                x[idx].clone(),
                p[idx].squeeze(),
                self.reference_checkpoint,
                ignore_bn=True,
            )
            # load checkpoint
            logging.info("load checkpoint model")
            self.reference_model.model.load_state_dict(checkpoint)
            watcher = ww.WeightWatcher(model=self.reference_model.model)
            details = watcher.analyze()
            summary = watcher.get_summary(details)
            # extract numbers
            out_tmp = torch.tensor([summary[kdx] for kdx in self.wwkey])
            out.append(out_tmp)

        return torch.stack(out)

    def forward_embeddings(self, x, p):
        return self.forward(x, p)
