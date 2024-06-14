# -*- coding: utf-8 -*-
################################################################################
# code originally take from https://github.com/Spijkervet/SimCLR/blob/master/modules/nt_xent.py
##########################

import torch
import torch.nn as nn

from einops import repeat

import warnings

from torchmetrics.functional import r2_score
from torchmetrics.functional import explained_variance


class MaskedReconLoss(nn.Module):
    """
    Recon loss with masks
    """

    def __init__(self, reduction):
        super(MaskedReconLoss, self).__init__()
        self.criterion = nn.MSELoss(reduction=reduction)
        self.loss_mean = None

    def forward(self, output, target, mask):
        """
        Args:
            output: torch.tensor of shape [batchsize,window,tokendim] with model predictions
            target: torch.tensor of shape [batchsize,window,tokendim] with model predictions
            mask: torch.tensor of shape [batchsize,window,tokendim] with maks for original non-zero values
        Returns:
            dict: "loss_recon": masekd MSE losss, "rsq": R^2 to mean
        """
        assert (
            output.shape == target.shape == mask.shape
        ), f"MSE loss error: prediction and target don't have the same shape. output {output.shape} vs target {target.shape} vs mask {mask.shape}"
        # apply mask
        output = mask * output
        loss = self.criterion(output, target)
        rsq = 0
        if self.loss_mean:
            rsq = torch.tensor(1 - loss.item() / self.loss_mean)
        else:
            rsq = explained_variance(
                preds=output, target=target, multioutput="uniform_average"
            )

            # rsq = r2_score(
            #     output.view(output.shape[0], -1),
            #     target.view(target.shape[0], -1),
            #     multioutput="uniform_average",
            # )
            # rsq = r2_score(output, target, multioutput="raw_values")

        return {"loss_recon": loss, "rsq": rsq}

    def set_mean_loss(self, data: torch.Tensor, mask: torch.Tensor):
        """
        #TODO
        """
        # check that data are tensor..
        assert isinstance(data, torch.Tensor)
        w_mean = data.mean(dim=0)  # compute over samples (dim0)
        # scale up to same size as data
        data_mean = repeat(w_mean, "l d -> n l d", n=data.shape[0])
        out_mean = self.forward(data_mean, data, mask)

        # compute mean
        print(f" mean loss: {out_mean['loss_recon']}")

        self.loss_mean = out_mean["loss_recon"]


################################################################################################
# contrastive loss
################################################################################################
class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        # create mask for negative samples: main diagonal, +-batch_size off-diagonal are set to 0
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        z_i, z_j: representations of batch in two different views. shape: batch_size x C
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        # dimension of similarity matrix
        N = 2 * self.batch_size
        # concat both representations to easily compute similarity matrix
        z = torch.cat((z_i, z_j), dim=0)
        # compute similarity matrix around dimension 2, which is the representation depth. the unsqueeze ensures the matmul/ outer product
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        # take positive samples
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        # We have 2N samples,resulting in: 2xNx1
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        # negative samples are singled out with the mask
        negative_samples = sim[self.mask].reshape(N, -1)

        # reformulate everything in terms of CrossEntropyLoss: https://pytorch.org/docs/master/generated/torch.nn.CrossEntropyLoss.html
        # labels in nominator, logits in denominator
        # positve class: 0 - that's the first component of the logits corresponding to the positive samples
        labels = torch.zeros(N).to(positive_samples.device).long()
        # the logits are NxN (N+1?) predictions for imaginary classes.
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss


class NT_Xent_pos(nn.Module):
    def __init__(self, batch_size, temperature):
        super(NT_Xent_pos, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.MSELoss(reduction="mean")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        # create mask for negative samples: main diagonal, +-batch_size off-diagonal are set to 0
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        z_i, z_j: representations of batch in two different views. shape: batch_size x C
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        # dimension of similarity matrix
        N = 2 * self.batch_size
        # concat both representations to easily compute similarity matrix
        z = torch.cat((z_i, z_j), dim=0)
        # compute similarity matrix around dimension 2, which is the representation depth. the unsqueeze ensures the matmul/ outer product
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        # take positive samples
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        # We have 2N samples,resulting in: 2xNx1
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        # negative samples are singled out with the mask
        # negative_samples = sim[self.mask].reshape(N, -1)

        # reformulate everything in terms of CrossEntropyLoss: https://pytorch.org/docs/master/generated/torch.nn.CrossEntropyLoss.html
        # labels in nominator, logits in denominator
        # positve class: 0 - that's the first component of the logits corresponding to the positive samples
        labels = torch.zeros(N).to(positive_samples.device).unsqueeze(dim=1)
        # just minimize the distance of positive samples to zero
        loss = self.criterion(positive_samples, labels)
        loss /= N
        return loss


################################################################################################
# contrastive + recon loss combination
################################################################################################
class GammaContrastReconLoss(nn.Module):
    """
    #TODO docstring
    Combines NTXent Loss with reconstruction loss.
    L = gamma*NTXentLoss + (1-gamma)*ReconstructionLoss
    """

    def __init__(
        self,
        gamma: float,
        reduction: str,
        batch_size: int,
        temperature: float,
        contrast="simclr",
        z_var_penalty: float = 0.0,
        z_norm_penalty: float = 0.0,
    ) -> None:
        super(GammaContrastReconLoss, self).__init__()
        # test for allowable gamma values
        assert 0 <= gamma <= 1
        self.gamma = gamma

        # z_var penalty
        self.z_var_penalty = z_var_penalty
        # z_norm penalty
        self.z_norm_penalty = z_norm_penalty

        # set contrast
        if contrast == "simclr":
            print("model: use simclr NT_Xent loss")
            self.loss_contrast = NT_Xent(batch_size, temperature)
        elif contrast == "positive":
            print("model: use only positive contrast loss")
            self.loss_contrast = NT_Xent_pos(batch_size, temperature)
        else:
            print("unrecognized contrast - use reconstruction only")

        self.loss_recon = MaskedReconLoss(
            reduction=reduction,
        )

        self.loss_mean = None

    def set_mean_loss(self, weights: torch.Tensor, mask=None) -> None:
        """
        Helper function to set mean loss in reconstruction loss
        """
        # if mask not set, set it to all ones
        if mask is None:
            mask = torch.ones(weights.shape)
        # call mean_loss function
        self.loss_recon.set_mean_loss(weights, mask=mask)

    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        m: torch.Tensor,
    ) -> dict:
        """
        Args:
            z_i, z_j are the two different views of the same batch encoded in the representation space. dim: batch_sizexrepresentation space
            y: reconstruction. dim: batch_sizexinput_size
            t: target dim: batch_sizexinput_size
            m: mask 1 where inputs are nonezero, 0 otherwise
        Returns:
            dict with "loss" as main aggregated loss key, as well as loss / rsq components
        """
        if self.gamma < 1e-10:
            out_recon = self.loss_recon(y, t, m)
            out = {
                "loss/loss": out_recon["loss_recon"],
                "loss/loss_contrast": torch.tensor(0.0),
                "loss/loss_recon": out_recon["loss_recon"],
            }
            for key in out_recon.keys():
                new_key = f"loss/{key}"
                if new_key not in out:
                    out[new_key] = out_recon[key]
        elif abs(1.0 - self.gamma) < 1e-10:
            loss_contrast = self.loss_contrast(z_i, z_j)
            out = {
                "loss/loss": loss_contrast,
                "loss/loss_contrast": loss_contrast,
                "loss/loss_recon": torch.tensor(0.0),
            }
        else:
            # combine loss components
            loss_contrast = self.loss_contrast(z_i, z_j)
            out_recon = self.loss_recon(y, t, m)
            loss = (
                self.gamma * loss_contrast + (1 - self.gamma) * out_recon["loss_recon"]
            )
            out = {
                "loss/loss": loss,
                "loss/loss_contrast": loss_contrast,
                "loss/loss_recon": out_recon["loss_recon"],
            }
            for key in out_recon.keys():
                new_key = f"loss/{key}"
                if new_key not in out:
                    out[new_key] = out_recon[key]
                    # compute embedding properties
        z_norm = torch.linalg.norm(z_i.view(z_i.shape[0], -1), ord=2, dim=1).mean()
        z_var = torch.mean(torch.var(z_i.view(z_i.shape[0], -1), dim=0))
        out["debug/z_norm"] = z_norm
        out["debug/z_var"] = z_var
        # if self.z_var_penalty > 0:
        out["loss/loss"] = out["loss/loss"] + self.z_var_penalty * z_var
        # if self.z_norm_penalty > 0:
        out["loss/loss"] = out["loss/loss"] + self.z_norm_penalty * z_norm

        return out
