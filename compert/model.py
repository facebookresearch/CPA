# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import json
import torch
import numpy as np

class NBLoss(torch.nn.Module):
    def __init__(self):
        super(NBLoss, self).__init__()

    def forward(self, yhat, y, eps=1e-8):
        """Negative binomial log-likelihood loss. It assumes targets `y` with n
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
        dim = yhat.size(1) // 2
        # means of the negative binomial (has to be positive support)
        mu = yhat[:, :dim]
        # inverse dispersion parameter (has to be positive support)
        theta = yhat[:, dim:]

        if theta.ndimension() == 1:
            # In this case, we reshape theta for broadcasting
            theta = theta.view(1, theta.size(0))
        t1 = torch.lgamma(theta + eps) + torch.lgamma(y + 1.0) -\
            torch.lgamma(y + theta + eps)
        t2 = (theta + y) * torch.log(1.0 + (mu / (theta + eps))) +\
            (y * (torch.log(theta + eps) - torch.log(mu + eps)))
        final = t1 + t2
        final = _nan2inf(final)

        return torch.mean(final)

def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)


class GaussianLoss(torch.nn.Module):
    """
    Gaussian log-likelihood loss. It assumes targets `y` with n rows and d
    columns, but estimates `yhat` with n rows and 2d columns. The columns 0:d
    of `yhat` contain estimated means, the columns d:2*d of `yhat` contain
    estimated variances. This module assumes that the estimated variances are
    positive---for numerical stability, it is recommended that the minimum
    estimated variance is greater than a small number (1e-3).
    """

    def __init__(self):
        super(GaussianLoss, self).__init__()

    def forward(self, yhat, y):
        dim = yhat.size(1) // 2
        mean = yhat[:, :dim]
        variance = yhat[:, dim:]

        term1 = variance.log().div(2)
        term2 = (y - mean).pow(2).div(variance.mul(2))

        return (term1 + term2).mean()


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
                if batch_norm and s < len(sizes) - 2 else None,
                torch.nn.ReLU()
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

    def __init__(self, dim, device, nonlin='sigmoid'):
        """Sigmoid modeling of continuous variable.
        Params
        ------
        nonlin : str (default: logsigm)
            One of logsigm, sigm.
        """
        super(GeneralizedSigmoid, self).__init__()
        self.nonlin = nonlin
        self.beta = torch.nn.Parameter(
            torch.ones(1, dim, device=device),
            requires_grad=True
        )
        self.bias = torch.nn.Parameter(
            torch.zeros(1, dim, device=device),
            requires_grad=True
        )

    def forward(self, x):
        if self.nonlin == 'logsigm':
            c0 = self.bias.sigmoid()
            return (torch.log1p(x) * self.beta + self.bias).sigmoid() - c0
        elif self.nonlin == 'sigm':
            c0 = self.bias.sigmoid()
            return (x * self.beta + self.bias).sigmoid() - c0
        else:
            return x

    def one_drug(self, x, i):
        if self.nonlin == 'logsigm':
            c0 = self.bias[0][i].sigmoid()
            return (torch.log1p(x) * self.beta[0][i] + self.bias[0][i]).sigmoid() - c0
        elif self.nonlin == 'sigm':
            c0 = self.bias[0][i].sigmoid()
            return (x * self.beta[0][i] + self.bias[0][i]).sigmoid() - c0
        else:
            return x

class ComPert(torch.nn.Module):
    """
    Our main module, the ComPert autoencoder
    """

    def __init__(
            self,
            num_genes,
            num_drugs,
            num_cell_types,
            device="cpu",
            seed=0,
            patience=5,
            loss_ae='gauss',
            doser_type='logsigm',
            decoder_activation='linear',
            hparams=""):
        super(ComPert, self).__init__()
        # set generic attributes
        self.num_genes = num_genes
        self.num_drugs = num_drugs
        self.num_cell_types = num_cell_types
        self.device = device
        self.seed = seed
        self.loss_ae = loss_ae
        # early-stopping
        self.patience = patience
        self.best_score = -1e3
        self.patience_trials = 0

        # set hyperparameters
        self.set_hparams_(seed, hparams)

        # set models
        self.encoder = MLP(
            [num_genes] +
            [self.hparams["autoencoder_width"]] *
            self.hparams["autoencoder_depth"] +
            [self.hparams["dim"]])

        self.decoder = MLP(
            [self.hparams["dim"]] +
            [self.hparams["autoencoder_width"]] *
            self.hparams["autoencoder_depth"] +
            [num_genes * 2], last_layer_act=decoder_activation)

        self.adversary_drugs = MLP(
            [self.hparams["dim"]] +
            [self.hparams["adversary_width"]] *
            self.hparams["adversary_depth"] +
            [num_drugs])

        self.adversary_cell_types = MLP(
            [self.hparams["dim"]] +
            [self.hparams["adversary_width"]] *
            self.hparams["adversary_depth"] +
            [num_cell_types])

        # set dosers
        self.doser_type = doser_type
        if doser_type == 'mlp':
            self.dosers = torch.nn.ModuleList()
            for _ in range(num_drugs):
                self.dosers.append(
                    MLP([1] +
                        [self.hparams["dosers_width"]] *
                        self.hparams["dosers_depth"] +
                        [1],
                        batch_norm=False))
        else:
            self.dosers = GeneralizedSigmoid(num_drugs, self.device,
            nonlin=doser_type)

        self.drug_embeddings = torch.nn.Embedding(
            num_drugs, self.hparams["dim"])
        self.cell_type_embeddings = torch.nn.Embedding(
            num_cell_types, self.hparams["dim"])

        # losses
        if self.loss_ae == 'nb':
            self.loss_autoencoder = NBLoss()
        else:
            self.loss_autoencoder = GaussianLoss()
        self.loss_adversary_drugs = torch.nn.BCEWithLogitsLoss()
        self.loss_adversary_cell_types = torch.nn.CrossEntropyLoss()
        self.iteration = 0

        self.to(self.device)

        # optimizers
        self.optimizer_autoencoder = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.drug_embeddings.parameters()) +
            list(self.cell_type_embeddings.parameters()),
            lr=self.hparams["autoencoder_lr"],
            weight_decay=self.hparams["autoencoder_wd"])

        self.optimizer_adversaries = torch.optim.Adam(
            list(self.adversary_drugs.parameters()) +
            list(self.adversary_cell_types.parameters()),
            lr=self.hparams["adversary_lr"],
            weight_decay=self.hparams["adversary_wd"])

        self.optimizer_dosers = torch.optim.Adam(
            self.dosers.parameters(),
            lr=self.hparams["dosers_lr"],
            weight_decay=self.hparams["dosers_wd"])

        # learning rate schedulers
        self.scheduler_autoencoder = torch.optim.lr_scheduler.StepLR(
            self.optimizer_autoencoder, step_size=self.hparams["step_size_lr"])

        self.scheduler_adversary = torch.optim.lr_scheduler.StepLR(
            self.optimizer_adversaries, step_size=self.hparams["step_size_lr"])

        self.scheduler_dosers = torch.optim.lr_scheduler.StepLR(
            self.optimizer_dosers, step_size=self.hparams["step_size_lr"])

        self.history = {'epoch': [], 'stats_epoch': []}

    def set_hparams_(self, seed, hparams):
        """
        Set hyper-parameters to (i) default values if `seed=0`, (ii) random
        values if `seed != 0`, or (iii) values fixed by user for those
        hyper-parameters specified in the JSON string `hparams`.
        """

        default = (seed == 0)
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.hparams = {
            "dim": 256 if default else
            int(np.random.choice([128, 256, 512])),
            "dosers_width": 64 if default else
            int(np.random.choice([32, 64, 128])),
            "dosers_depth": 2 if default else
            int(np.random.choice([1, 2, 3])),
            "dosers_lr": 1e-3 if default else
            float(10**np.random.uniform(-4, -2)),
            "dosers_wd": 1e-7 if default else
            float(10**np.random.uniform(-8, -5)),
            "autoencoder_width": 512 if default else
            int(np.random.choice([256, 512, 1024])),
            "autoencoder_depth": 4 if default else
            int(np.random.choice([3, 4, 5])),
            "adversary_width": 128 if default else
            int(np.random.choice([64, 128, 256])),
            "adversary_depth": 3 if default else
            int(np.random.choice([2, 3, 4])),
            "reg_adversary": 5 if default else
            float(10**np.random.uniform(-2, 2)),
            "penalty_adversary": 3 if default else
            float(10**np.random.uniform(-2, 1)),
            "autoencoder_lr": 1e-3 if default else
            float(10**np.random.uniform(-4, -2)),
            "adversary_lr": 3e-4 if default else
            float(10**np.random.uniform(-5, -3)),
            "autoencoder_wd": 1e-6 if default else
            float(10**np.random.uniform(-8, -4)),
            "adversary_wd": 1e-4 if default else
            float(10**np.random.uniform(-6, -3)),
            "adversary_steps": 3 if default else
            int(np.random.choice([1, 2, 3, 4, 5])),
            "batch_size": 128 if default else
            int(np.random.choice([64, 128, 256, 512])),
            "step_size_lr": 45 if default else
            int(np.random.choice([15, 25, 45])),
        }

        # the user may fix some hparams
        if hparams != "":
            if isinstance(hparams, str):
                self.hparams.update(json.loads(hparams))
            else:
                self.hparams.update(hparams)

        return self.hparams

    def move_inputs_(self, genes, drugs, cell_types):
        """
        Move minibatch tensors to CPU/GPU.
        """
        if genes.device.type != self.device:
            genes = genes.to(self.device)
            drugs = drugs.to(self.device)
            cell_types = cell_types.to(self.device)
        return genes, drugs, cell_types

    def compute_drug_embeddings_(self, drugs):
        """
        Compute sum of drug embeddings, each of them multiplied by its
        dose-response curve.
        """

        if self.doser_type == 'mlp':
            doses = []
            for d in range(drugs.size(1)):
                this_drug = drugs[:, d].view(-1, 1)
                doses.append(self.dosers[d](this_drug).sigmoid() * this_drug.gt(0))
            return torch.cat(doses, 1) @ self.drug_embeddings.weight
        else:
            return self.dosers(drugs) @ self.drug_embeddings.weight


    def predict(self, genes, drugs, cell_types, return_latent_basal=False):
        """
        Predict "what would have the gene expression `genes` been, had the
        cells in `genes` with cell types `cell_types` been treated with
        combination of drugs `drugs`.
        """

        genes, drugs, cell_types = self.move_inputs_(genes, drugs, cell_types)

        latent_basal = self.encoder(genes)
        drug_emb = self.compute_drug_embeddings_(drugs)
        cell_emb = self.cell_type_embeddings(cell_types.argmax(1))

        latent_treated = latent_basal + drug_emb + cell_emb
        gene_reconstructions = self.decoder(latent_treated)

        # convert variance estimates to a positive value in [1e-3, \infty)
        dim = gene_reconstructions.size(1) // 2
        gene_reconstructions[:, dim:] =\
            gene_reconstructions[:, dim:].exp().add(1).log().add(1e-3)

        if self.loss_ae == 'nb':
            gene_reconstructions[:, :dim] =\
                gene_reconstructions[:, :dim].exp().add(1).log().add(1e-4)
            # gene_reconstructions[:, :dim] = torch.clamp(gene_reconstructions[:, :dim], min=1e-4, max=1e4)
            # gene_reconstructions[:, dim:] = torch.clamp(gene_reconstructions[:, dim:], min=1e-6, max=1e6)


        if return_latent_basal:
            return gene_reconstructions, latent_basal

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

    def update(self, genes, drugs, cell_types):
        """
        Update ComPert's parameters given a minibatch of genes, drugs, and
        cell types.
        """

        genes, drugs, cell_types = self.move_inputs_(genes, drugs, cell_types)

        gene_reconstructions, latent_basal = self.predict(
            genes, drugs, cell_types, return_latent_basal=True)

        reconstruction_loss = self.loss_autoencoder(
            gene_reconstructions, genes)

        adversary_drugs_predictions = self.adversary_drugs(
            latent_basal)
        adversary_drugs_loss = self.loss_adversary_drugs(
            adversary_drugs_predictions, drugs.gt(0).float())

        adversary_cell_types_predictions = self.adversary_cell_types(
            latent_basal)
        adversary_cell_types_loss = self.loss_adversary_cell_types(
            adversary_cell_types_predictions, cell_types.argmax(1))

        # two place-holders for when adversary is not executed
        adversary_drugs_penalty = torch.Tensor([0])
        adversary_cell_types_penalty = torch.Tensor([0])

        if self.iteration % self.hparams["adversary_steps"]:
            adversary_drugs_penalty = torch.autograd.grad(
                adversary_drugs_predictions.sum(),
                latent_basal,
                create_graph=True)[0].pow(2).mean()

            adversary_cell_types_penalty = torch.autograd.grad(
                adversary_cell_types_predictions.sum(),
                latent_basal,
                create_graph=True)[0].pow(2).mean()

            self.optimizer_adversaries.zero_grad()
            (adversary_drugs_loss +
             self.hparams["penalty_adversary"] *
             adversary_drugs_penalty +
             adversary_cell_types_loss +
             self.hparams["penalty_adversary"] *
             adversary_cell_types_penalty).backward()
            self.optimizer_adversaries.step()
        else:
            self.optimizer_autoencoder.zero_grad()
            self.optimizer_dosers.zero_grad()
            (reconstruction_loss -
             self.hparams["reg_adversary"] *
             adversary_drugs_loss -
             self.hparams["reg_adversary"] *
             adversary_cell_types_loss).backward()
            self.optimizer_autoencoder.step()
            self.optimizer_dosers.step()

        self.iteration += 1

        return {
            "loss_reconstruction": reconstruction_loss.item(),
            "loss_adv_drugs": adversary_drugs_loss.item(),
            "loss_adv_cell_types": adversary_cell_types_loss.item(),
            "penalty_adv_drugs": adversary_drugs_penalty.item(),
            "penalty_adv_cell_types": adversary_cell_types_penalty.item()
        }

    @classmethod
    def defaults(self):
        """
        Returns the list of default hyper-parameters for ComPert
        """

        return self.set_hparams_(self, 0, "")
