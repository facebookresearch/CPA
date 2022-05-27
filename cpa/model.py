# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from http.client import RemoteDisconnected
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

class NBLoss(torch.nn.Module):
    def __init__(self):
        super(NBLoss, self).__init__()

    def forward(self, mu, y, theta, eps=1e-8):
        """Negative binomial negative log-likelihood. It assumes targets `y` with n
        rows and d columns, but estimates `yhat` with n rows and 2d columns.
        The columns 0:d of `yhat` contain estimated means, the columns d:2*d of
        `yhat` contain estimated variances. This module assumes that the
        estimated mean and inverse dispersion are positive---for numerical
        stability, it is recommended that the minimum estimated variance is
        greater than a small number (1e-3).
        Parameters
        ----------
        yhat: Tensor
                Torch Tensor of reeconstructed data.
        y: Tensor
                Torch Tensor of ground truth data.
        eps: Float
                numerical stability constant.
        """
        if theta.ndimension() == 1:
            # In this case, we reshape theta for broadcasting
            theta = theta.view(1, theta.size(0))
        log_theta_mu_eps = torch.log(theta + mu + eps)
        res = (
            theta * (torch.log(theta + eps) - log_theta_mu_eps)
            + y * (torch.log(mu + eps) - log_theta_mu_eps)
            + torch.lgamma(y + theta)
            - torch.lgamma(theta)
            - torch.lgamma(y + 1)
        )
        res = _nan2inf(res)
        return -torch.mean(res)
    
def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)

class MLP(torch.nn.Module):
    """
    A multilayer perceptron with ReLU activations and optional BatchNorm.
    """

    def __init__(self, sizes, batch_norm=True, last_layer_act="linear"):
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers += [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 2
                else None,
                torch.nn.ReLU(),
            ]

        layers = [l for l in layers if l is not None][:-1]
        self.activation = last_layer_act
        if self.activation == "linear":
            pass
        elif self.activation == "ReLU":
            self.relu = torch.nn.ReLU()
        else:
            raise ValueError("last_layer_act must be one of 'linear' or 'ReLU'")

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        if self.activation == "ReLU":
            x = self.network(x)
            dim = x.size(1) // 2
            return torch.cat((self.relu(x[:, :dim]), x[:, dim:]), dim=1)
        return self.network(x)


class GeneralizedSigmoid(torch.nn.Module):
    """
    Sigmoid, log-sigmoid or linear functions for encoding dose-response for
    drug perurbations.
    """

    def __init__(self, dim, device, nonlin="sigmoid"):
        """Sigmoid modeling of continuous variable.
        Params
        ------
        nonlin : str (default: logsigm)
            One of logsigm, sigm.
        """
        super(GeneralizedSigmoid, self).__init__()
        self.nonlin = nonlin
        self.beta = torch.nn.Parameter(
            torch.ones(1, dim, device=device), requires_grad=True
        )
        self.bias = torch.nn.Parameter(
            torch.zeros(1, dim, device=device), requires_grad=True
        )

    def forward(self, x):
        if self.nonlin == "logsigm":
            c0 = self.bias.sigmoid()
            return (torch.log1p(x) * self.beta + self.bias).sigmoid() - c0
        elif self.nonlin == "sigm":
            c0 = self.bias.sigmoid()
            return (x * self.beta + self.bias).sigmoid() - c0
        else:
            return x

    def one_drug(self, x, i):
        if self.nonlin == "logsigm":
            c0 = self.bias[0][i].sigmoid()
            return (torch.log1p(x) * self.beta[0][i] + self.bias[0][i]).sigmoid() - c0
        elif self.nonlin == "sigm":
            c0 = self.bias[0][i].sigmoid()
            return (x * self.beta[0][i] + self.bias[0][i]).sigmoid() - c0
        else:
            return x


class CPA(torch.nn.Module):
    """
    Our main module, the CPA autoencoder
    """

    def __init__(
        self,
        num_genes,
        num_drugs,
        num_covariates,
        device="cuda",
        seed=0,
        patience=5,
        loss_ae="gauss",
        doser_type="mlp",
        decoder_activation="linear",
        hparams="",
    ):
        super(CPA, self).__init__()
        # set generic attributes
        self.num_genes = num_genes
        self.num_drugs = num_drugs
        self.num_covariates = num_covariates
        self.device = device
        self.seed = seed
        self.loss_ae = loss_ae
        # early-stopping
        self.patience = patience
        self.best_score = -1e3
        self.patience_trials = 0

        # set hyperparameters
        self.set_hparams_(hparams)

        # set models
        self.encoder = MLP(
            [num_genes]
            + [self.hparams["autoencoder_width"]] * self.hparams["autoencoder_depth"]
            + [self.hparams["dim"]]
        )

        self.decoder = MLP(
            [self.hparams["dim"]]
            + [self.hparams["autoencoder_width"]] * self.hparams["autoencoder_depth"]
            + [num_genes * 2],
            last_layer_act=decoder_activation,
        )

        self.adversary_drugs = MLP(
            [self.hparams["dim"]]
            + [self.hparams["adversary_width"]] * self.hparams["adversary_depth"]
            + [num_drugs]
        )

        self.loss_adversary_drugs = torch.nn.BCEWithLogitsLoss()
        self.doser_type = doser_type
        if doser_type == "mlp":
            self.dosers = torch.nn.ModuleList()
            for _ in range(num_drugs):
                self.dosers.append(
                    MLP(
                        [1]
                        + [self.hparams["dosers_width"]] * self.hparams["dosers_depth"]
                        + [1],
                        batch_norm=False,
                    )
                )
        else:
            self.dosers = GeneralizedSigmoid(num_drugs, self.device, nonlin=doser_type)

        if self.num_covariates == [0]:
            pass
        else:
            assert 0 not in self.num_covariates
            self.adversary_covariates = []
            self.loss_adversary_covariates = []
            self.covariates_embeddings = (
                []
            )  # TODO: Continue with checking that dict assignment is possible via covaraites names and if dict are possible to use in optimisation
            for num_covariate in self.num_covariates:
                self.adversary_covariates.append(
                    MLP(
                        [self.hparams["dim"]]
                        + [self.hparams["adversary_width"]]
                        * self.hparams["adversary_depth"]
                        + [num_covariate]
                    )
                )
                self.loss_adversary_covariates.append(torch.nn.CrossEntropyLoss())
                self.covariates_embeddings.append(
                    torch.nn.Embedding(num_covariate, self.hparams["dim"])
                )
            self.covariates_embeddings = torch.nn.Sequential(
                *self.covariates_embeddings
            )

        self.drug_embeddings = torch.nn.Embedding(self.num_drugs, self.hparams["dim"])
        # losses
        if self.loss_ae == "nb":
            self.loss_autoencoder = NBLoss()
        elif self.loss_ae == 'gauss':
            self.loss_autoencoder = nn.GaussianNLLLoss()

        self.iteration = 0

        self.to(self.device)

        # optimizers
        has_drugs = self.num_drugs > 0
        has_covariates = self.num_covariates[0] > 0
        get_params = lambda model, cond: list(model.parameters()) if cond else []
        _parameters = (
            get_params(self.encoder, True)
            + get_params(self.decoder, True)
            + get_params(self.drug_embeddings, has_drugs)
        )
        for emb in self.covariates_embeddings:
            _parameters.extend(get_params(emb, has_covariates))

        self.optimizer_autoencoder = torch.optim.Adam(
            _parameters,
            lr=self.hparams["autoencoder_lr"],
            weight_decay=self.hparams["autoencoder_wd"],
        )

        _parameters = get_params(self.adversary_drugs, has_drugs)
        for adv in self.adversary_covariates:
            _parameters.extend(get_params(adv, has_covariates))

        self.optimizer_adversaries = torch.optim.Adam(
            _parameters,
            lr=self.hparams["adversary_lr"],
            weight_decay=self.hparams["adversary_wd"],
        )

        if has_drugs:
            self.optimizer_dosers = torch.optim.Adam(
                self.dosers.parameters(),
                lr=self.hparams["dosers_lr"],
                weight_decay=self.hparams["dosers_wd"],
            )

        # learning rate schedulers
        self.scheduler_autoencoder = torch.optim.lr_scheduler.StepLR(
            self.optimizer_autoencoder, step_size=self.hparams["step_size_lr"]
        )

        self.scheduler_adversary = torch.optim.lr_scheduler.StepLR(
            self.optimizer_adversaries, step_size=self.hparams["step_size_lr"]
        )

        if has_drugs:
            self.scheduler_dosers = torch.optim.lr_scheduler.StepLR(
                self.optimizer_dosers, step_size=self.hparams["step_size_lr"]
            )

        self.history = {"epoch": [], "stats_epoch": []}

    def set_hparams_(self, hparams):
        """
        Set hyper-parameters to default values or values fixed by user for those
        hyper-parameters specified in the JSON string `hparams`.
        """

        self.hparams = {
            "dim": 128,
            "dosers_width": 128,
            "dosers_depth": 2,
            "dosers_lr": 4e-3,
            "dosers_wd": 1e-7,
            "autoencoder_width": 128,
            "autoencoder_depth": 3,
            "adversary_width": 64,
            "adversary_depth": 2,
            "reg_adversary": 60,
            "penalty_adversary": 60,
            "autoencoder_lr": 3e-4,
            "adversary_lr": 3e-4,
            "autoencoder_wd": 4e-7,
            "adversary_wd": 4e-7,
            "adversary_steps": 3,
            "batch_size": 256,
            "step_size_lr": 45,
        }

        # the user may fix some hparams
        if hparams != "":
            if isinstance(hparams, str):
                self.hparams.update(json.loads(hparams))
            else:
                self.hparams.update(hparams)

        return self.hparams

    def move_inputs_(self, genes, drugs, covariates):
        """
        Move minibatch tensors to CPU/GPU.
        """
        if genes.device.type != self.device:
            genes = genes.to(self.device)
            if drugs is not None:
                drugs = drugs.to(self.device)
            if covariates is not None:
                covariates = [cov.to(self.device) for cov in covariates]
        return (genes, drugs, covariates)

    def compute_drug_embeddings_(self, drugs):
        """
        Compute sum of drug embeddings, each of them multiplied by its
        dose-response curve.
        """

        if self.doser_type == "mlp":
            doses = []
            for d in range(drugs.size(1)):
                this_drug = drugs[:, d].view(-1, 1)
                doses.append(self.dosers[d](this_drug).sigmoid() * this_drug.gt(0))
            return torch.cat(doses, 1) @ self.drug_embeddings.weight
        else:
            return self.dosers(drugs) @ self.drug_embeddings.weight

    def predict(
        self, 
        genes, 
        drugs, 
        covariates, 
        return_latent_basal=False,
        return_latent_treated=False,
    ):
        """
        Predict "what would have the gene expression `genes` been, had the
        cells in `genes` with cell types `cell_types` been treated with
        combination of drugs `drugs`.
        """

        genes, drugs, covariates = self.move_inputs_(genes, drugs, covariates)

        latent_basal = self.encoder(genes)

        latent_treated = latent_basal

        if self.num_drugs > 0:
            latent_treated = latent_treated + self.compute_drug_embeddings_(drugs)
        if self.num_covariates[0] > 0:
            for i, emb in enumerate(self.covariates_embeddings):
                emb = emb.to(self.device)
                latent_treated = latent_treated + emb(
                    covariates[i].argmax(1)
                )  #argmax because OHE

        gene_reconstructions = self.decoder(latent_treated)
        dim = gene_reconstructions.size(1) // 2
        if self.loss_ae == 'gauss':
            # convert variance estimates to a positive value in [1e-3, \infty)
            gene_means = gene_reconstructions[:, :dim]
            gene_vars = F.softplus(gene_reconstructions[:, dim:]).add(1e-3)
            #gene_vars = gene_reconstructions[:, dim:].exp().add(1).log().add(1e-3)
            gene_reconstructions = torch.cat([gene_means, gene_vars], dim=1)

        if self.loss_ae == 'nb':
            gene_mus = F.softplus(gene_reconstructions[:, :dim]).add(1e-3)
            gene_thetas = F.softplus(gene_reconstructions[:, dim:]).add(1e-3)
            #gene_reconstructions[:, :dim] = torch.clamp(gene_reconstructions[:, :dim], min=1e-4, max=1e4)
            #gene_reconstructions[:, dim:] = torch.clamp(gene_reconstructions[:, dim:], min=1e-4, max=1e4)
            gene_reconstructions = torch.cat([gene_mus, gene_thetas], dim=1)

        if return_latent_basal:
            if return_latent_treated:
                return gene_reconstructions, latent_basal, latent_treated
            else:
                return gene_reconstructions, latent_basal
        if return_latent_treated:
            return gene_reconstructions, latent_treated
        return gene_reconstructions

    def early_stopping(self, score):
        """
        Decays the learning rate, and possibly early-stops training.
        """
        self.scheduler_autoencoder.step()
        self.scheduler_adversary.step()
        self.scheduler_dosers.step()

        if score > self.best_score:
            self.best_score = score
            self.patience_trials = 0
        else:
            self.patience_trials += 1

        return self.patience_trials > self.patience

    def update(self, genes, drugs, covariates):
        """
        Update CPA's parameters given a minibatch of genes, drugs, and
        cell types.
        """
        genes, drugs, covariates = self.move_inputs_(genes, drugs, covariates)
        gene_reconstructions, latent_basal = self.predict(
            genes,
            drugs,
            covariates,
            return_latent_basal=True,
        )

        dim = gene_reconstructions.size(1) // 2
        reconstruction_loss = self.loss_autoencoder(gene_reconstructions[:, :dim], genes, gene_reconstructions[:, dim:])
        adversary_drugs_loss = torch.tensor([0.0], device=self.device)
        if self.num_drugs > 0:
            adversary_drugs_predictions = self.adversary_drugs(latent_basal)
            adversary_drugs_loss = self.loss_adversary_drugs(
                adversary_drugs_predictions, drugs.gt(0).float()
            )

        adversary_covariates_loss = torch.tensor(
            [0.0], device=self.device
        )
        if self.num_covariates[0] > 0:
            adversary_covariate_predictions = []
            for i, adv in enumerate(self.adversary_covariates):
                adv = adv.to(self.device)
                adversary_covariate_predictions.append(adv(latent_basal))
                adversary_covariates_loss += self.loss_adversary_covariates[i](
                    adversary_covariate_predictions[-1], covariates[i].argmax(1)
                )

        # two place-holders for when adversary is not executed
        adversary_drugs_penalty = torch.tensor([0.0], device=self.device)
        adversary_covariates_penalty = torch.tensor([0.0], device=self.device)

        if self.iteration % self.hparams["adversary_steps"]:

            def compute_gradients(output, input):
                grads = torch.autograd.grad(output, input, create_graph=True)
                grads = grads[0].pow(2).mean()
                return grads

            if self.num_drugs > 0:
                adversary_drugs_penalty = compute_gradients(
                    adversary_drugs_predictions.sum(), latent_basal
                )

            if self.num_covariates[0] > 0:
                adversary_covariates_penalty = torch.tensor([0.0], device=self.device)
                for pred in adversary_covariate_predictions:
                    adversary_covariates_penalty += compute_gradients(
                        pred.sum(), latent_basal
                    )  # TODO: Adding up tensor sum, is that right?

            self.optimizer_adversaries.zero_grad()
            (
                adversary_drugs_loss
                + self.hparams["penalty_adversary"] * adversary_drugs_penalty
                + adversary_covariates_loss
                + self.hparams["penalty_adversary"] * adversary_covariates_penalty
            ).backward()
            self.optimizer_adversaries.step()
        else:
            self.optimizer_autoencoder.zero_grad()
            if self.num_drugs > 0:
                self.optimizer_dosers.zero_grad()
            (
                reconstruction_loss
                - self.hparams["reg_adversary"] * adversary_drugs_loss
                - self.hparams["reg_adversary"] * adversary_covariates_loss
            ).backward()
            self.optimizer_autoencoder.step()
            if self.num_drugs > 0:
                self.optimizer_dosers.step()
        self.iteration += 1

        return {
            "loss_reconstruction": reconstruction_loss.item(),
            "loss_adv_drugs": adversary_drugs_loss.item(),
            "loss_adv_covariates": adversary_covariates_loss.item(),
            "penalty_adv_drugs": adversary_drugs_penalty.item(),
            "penalty_adv_covariates": adversary_covariates_penalty.item(),
        }

    @classmethod
    def defaults(self):
        """
        Returns the list of default hyper-parameters for CPA
        """

        return self.set_hparams_(self, "")
