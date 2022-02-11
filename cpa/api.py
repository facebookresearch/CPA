import copy
import itertools
import os
import pprint
import time
from collections import defaultdict
from typing import Optional, Union, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.distributions import (
    NegativeBinomial,
    Normal
)
from cpa.train import evaluate, prepare_cpa
from cpa.helper import _convert_mean_disp_to_counts_logits
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from tqdm import tqdm

class API:
    """
    API for CPA model to make it compatible with scanpy.
    """

    def __init__(
        self,
        data,
        covariate_keys=["cell_type"],
        split_key="split",
        perturbation_key="condition",
        dose_key="dose_val",
        doser_type="mlp",
        decoder_activation="linear",
        loss_ae="gauss",
        patience=200,
        seed=0,
        pretrained=None,
        device="cuda",
        save_dir="/tmp/",  # directory to save the model
        hparams={},
        only_parameters=False,
        control=None,
    ):
        """
        Parameters
        ----------
        data : str or `AnnData`
            AnndData object or a full path to the file in the .h5ad format.
        covariate_keys : list (default: ['cell_type'])
            List of names in the .obs of AnnData that should be used as
            covariates. #TODO handel empty list.
        split_key : str (default: 'split')
            Name of the column in .obs of AnnData to use for splitting the
            dataset into train, test and validation.
        perturbation_key : str (default: 'condition')
            Name of the column in .obs of AnnData to use for perturbation
            variable.
        dose_key : str (default: 'dose_val')
            Name of the column in .obs of AnnData to use for continious
            covariate.
        doser_type : str (default: 'logsigm')
            Type of the nonlinearity in the latent space for the continious
            covariate encoding: sigm, logsigm, mlp.
        decoder_activation : str (default: 'linear')
            Last layer of the decoder.
        loss_ae : str (default: 'gauss')
            Loss (currently only gaussian loss is supported).
        patience : int (default: 20)
            Patience for early stopping.
        seed : int (default: 20)
            Random seed.
        sweep_seeds : int (default: 0)
            Random seed. #TODO check  if I can remove it.
        pretrained : str (default: None)
            Full path to the pretrained model.
        save_dir : str (default: '/tmp/')
            Folder to save the model.
        device : str (default: 'cpu')
            Device for model computations. If None, will try to use CUDA if
            available.
        hparams : dict (default: {})
            Parameters for the architecture of the CPA model.
        """
        args = locals()
        del args["self"]

        if not (pretrained is None):
            state, self.used_args, self.history = torch.load(
                pretrained, map_location=torch.device(device)
            )
            self.args = self.used_args
            self.args["data"] = data
            self.args["covariate_keys"] = covariate_keys
            self.args["device"] = device
            self.args["control"] = control
            if only_parameters:
                state = None
                print(f"Loaded ARGS of the model from:\t{pretrained}")
            else:
                print(f"Loaded pretrained model from:\t{pretrained}")
        else:
            state = None
            self.args = args

        # pprint.pprint(self.args)
        self.model, self.datasets = prepare_cpa(self.args, state_dict=state)
        if not (pretrained is None) and (not only_parameters):
            self.model.history = self.history
        self.args["save_dir"] = save_dir
        self.args["hparams"] = self.model.hparams

        if not (save_dir is None):
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

        dataset = self.datasets["training"]
        self.perturbation_key = dataset.perturbation_key
        self.dose_key = dataset.dose_key
        self.covariate_keys = covariate_keys  # very important, specifies the order of
        # covariates during training
        self.min_dose = dataset.drugs[dataset.drugs > 0].min().item()
        self.max_dose = dataset.drugs[dataset.drugs > 0].max().item()

        self.var_names = dataset.var_names

        self.unique_perts = list(dataset.perts_dict.keys())

        self.unique_covars = {}
        for cov in dataset.covars_dict:
            self.unique_covars[cov] = list(dataset.covars_dict[cov].keys())
        self.num_drugs = dataset.num_drugs

        self.perts_dict = dataset.perts_dict
        self.covars_dict = dataset.covars_dict

        self.drug_ohe = torch.Tensor(list(dataset.perts_dict.values()))

        self.covars_ohe = {}
        for cov in dataset.covars_dict:
            self.covars_ohe[cov] = torch.LongTensor(
                list(dataset.covars_dict[cov].values())
            )

        self.emb_covars = {}
        for cov in dataset.covars_dict:
            self.emb_covars[cov] = None
        self.emb_perts = None
        self.seen_covars_perts = None
        self.comb_emb = None
        self.control_cat = None

        self.seen_covars_perts = {}
        for k in self.datasets.keys():
            self.seen_covars_perts[k] = np.unique(self.datasets[k].pert_categories)

        self.measured_points = {}
        self.num_measured_points = {}
        for k in self.datasets.keys():
            self.measured_points[k] = {}
            self.num_measured_points[k] = {}
            for pert in np.unique(self.datasets[k].pert_categories):
                num_points = len(np.where(self.datasets[k].pert_categories == pert)[0])
                self.num_measured_points[k][pert] = num_points

                *cov_list, drug, dose = pert.split("_")
                cov = "_".join(cov_list)
                if not ("+" in dose):
                    dose = float(dose)
                if cov in self.measured_points[k].keys():
                    if drug in self.measured_points[k][cov].keys():
                        self.measured_points[k][cov][drug].append(dose)
                    else:
                        self.measured_points[k][cov][drug] = [dose]
                else:
                    self.measured_points[k][cov] = {drug: [dose]}

        self.measured_points["all"] = copy.deepcopy(self.measured_points["training"])
        for cov in self.measured_points["ood"].keys():
            for pert in self.measured_points["ood"][cov].keys():
                if pert in self.measured_points["training"][cov].keys():
                    self.measured_points["all"][cov][pert] = (
                        self.measured_points["training"][cov][pert].copy()
                        + self.measured_points["ood"][cov][pert].copy()
                    )
                else:
                    self.measured_points["all"][cov][pert] = self.measured_points[
                        "ood"
                    ][cov][pert].copy()

    def load_from_old(self, pretrained):
        """
        Parameters
        ----------
        pretrained : str
            Full path to the pretrained model.
        """
        print(f"Loaded pretrained model from:\t{pretrained}")
        state, self.used_args, self.history = torch.load(
            pretrained, map_location=torch.device(self.args["device"])
        )
        self.model.load_state_dict(state_dict)
        self.model.history = self.history

    def print_args(self):
        pprint.pprint(self.args)

    def load(self, pretrained):
        """
        Parameters
        ----------
        pretrained : str
            Full path to the pretrained model.
        """  # TODO fix compatibility
        print(f"Loaded pretrained model from:\t{pretrained}")
        state, self.used_args, self.history = torch.load(
            pretrained, map_location=torch.device(self.args["device"])
        )
        self.model.load_state_dict(state_dict)

    def train(
        self,
        max_epochs=1,
        checkpoint_freq=20,
        run_eval=False,
        max_minutes=60,
        filename="model.pt",
        batch_size=None,
        save_dir=None,
        seed=0,
    ):
        """
        Parameters
        ----------
        max_epochs : int (default: 1)
            Maximum number epochs for training.
        checkpoint_freq : int (default: 20)
            Checkoint frequencty to save intermediate results.
        run_eval : bool (default: False)
            Whether or not to run disentanglement and R2 evaluation during training.
        max_minutes : int (default: 60)
            Maximum computation time in minutes.
        filename : str (default: 'model.pt')
            Name of the file without the directoty path to save the model.
            Name should be with .pt extension.
        batch_size : int, optional (default: None)
            Batch size for training. If None, uses default batch size specified
            in hparams.
        save_dir : str, optional (default: None)
            Full path to the folder to save the model. If None, will use from
            the path specified during init.
        seed : int (default: None)
            Random seed. If None, uses default random seed specified during init.
        """
        args = locals()
        del args["self"]
        # print('Training...')
        # pprint.pprint(args)
        #

        if batch_size is None:
            batch_size = self.model.hparams["batch_size"]
            args["batch_size"] = batch_size
            self.args["batch_size"] = batch_size

        if save_dir is None:
            save_dir = self.args["save_dir"]
        print("Results will be saved to the folder:", save_dir)

        self.datasets.update(
            {
                "loader_tr": torch.utils.data.DataLoader(
                    self.datasets["training"], batch_size=batch_size, shuffle=True
                )
            }
        )

        # pjson({"training_args": args})
        # pjson({"autoencoder_params": self.model.hparams})
        self.model.train()

        start_time = time.time()
        pbar = tqdm(range(max_epochs), ncols=80)
        try:
            for epoch in pbar:
                epoch_training_stats = defaultdict(float)

                for data in self.datasets["loader_tr"]:
                    genes, drugs, covariates = data[0], data[1], data[2:]
                    minibatch_training_stats = self.model.update(
                        genes, drugs, covariates
                    )

                    for key, val in minibatch_training_stats.items():
                        epoch_training_stats[key] += val

                for key, val in epoch_training_stats.items():
                    epoch_training_stats[key] = val / len(self.datasets["loader_tr"])
                    if not (key in self.model.history.keys()):
                        self.model.history[key] = []
                    self.model.history[key].append(epoch_training_stats[key])
                self.model.history["epoch"].append(epoch)

                ellapsed_minutes = (time.time() - start_time) / 60
                self.model.history["elapsed_time_min"] = ellapsed_minutes

                # decay learning rate if necessary
                # also check stopping condition: patience ran out OR
                # time ran out OR max epochs achieved
                stop = ellapsed_minutes > max_minutes or (epoch == max_epochs - 1)

                pbar.set_description(
                    f"Rec: {epoch_training_stats['loss_reconstruction']:.4f}, "
                    + f"AdvPert: {epoch_training_stats['loss_adv_drugs']:.2f}, "
                    + f"AdvCov: {epoch_training_stats['loss_adv_covariates']:.2f}"
                )

                if (epoch % checkpoint_freq) == 0 or stop:
                    if run_eval == True:
                        evaluation_stats = evaluate(self.model, self.datasets)
                        for key, val in evaluation_stats.items():
                            if not (key in self.model.history.keys()):
                                self.model.history[key] = []
                            self.model.history[key].append(val)
                        self.model.history["stats_epoch"].append(epoch)
                        stop = stop or self.model.early_stopping(
                            np.mean(evaluation_stats["test"])
                        )
                    else:
                        stop = stop or self.model.early_stopping(
                            np.mean(epoch_training_stats["test"])
                        )
                        evaluation_stats = None

                    if stop:
                        self.save(f"{save_dir}{filename}")
                        pprint.pprint(
                            {
                                "epoch": epoch,
                                "training_stats": epoch_training_stats,
                                "evaluation_stats": evaluation_stats,
                                "ellapsed_minutes": ellapsed_minutes,
                            }
                        )

                        print(f"Stop epoch: {epoch}")
                        break


        except KeyboardInterrupt:
            self.save(f"{save_dir}{filename}")

        self.save(f"{save_dir}{filename}")

    def save(self, filename):
        """
        Parameters
        ----------
        filename : str
            Full path to save pretrained model.
        """
        torch.save((self.model.state_dict(), self.args, self.model.history), filename)
        self.history = self.model.history
        print(f"Model saved to: {filename}")

    def _init_pert_embeddings(self):
        dose = 1.0
        self.emb_perts = (
            self.model.compute_drug_embeddings_(
                dose * self.drug_ohe.to(self.model.device)
            )
            .cpu()
            .clone()
            .detach()
            .numpy()
        )

    def get_drug_embeddings(self, dose=1.0, return_anndata=True):
        """
        Parameters
        ----------
        dose : int (default: 1.0)
            Dose at which to evaluate latent embedding vector.
        return_anndata : bool, optional (default: True)
            Return embedding wrapped into anndata object.

        Returns
        -------
        If return_anndata is True, returns anndata object. Otherwise, doesn't
        return anything. Always saves embeddding in self.emb_perts.
        """
        self._init_pert_embeddings()

        emb_perts = (
            self.model.compute_drug_embeddings_(
                dose * self.drug_ohe.to(self.model.device)
            )
            .cpu()
            .clone()
            .detach()
            .numpy()
        )

        if return_anndata:
            adata = sc.AnnData(emb_perts)
            adata.obs[self.perturbation_key] = self.unique_perts
            return adata

    def _init_covars_embeddings(self):
        combo_list = []
        for covars_key in self.covariate_keys:
            combo_list.append(self.unique_covars[covars_key])
            if self.emb_covars[covars_key] is None:
                i_cov = self.covariate_keys.index(covars_key)
                self.emb_covars[covars_key] = dict(
                    zip(
                        self.unique_covars[covars_key],
                        self.model.covariates_embeddings[i_cov](
                            self.covars_ohe[covars_key].to(self.model.device).argmax(1)
                        )
                        .cpu()
                        .clone()
                        .detach()
                        .numpy(),
                    )
                )
        self.emb_covars_combined = {}
        for combo in list(itertools.product(*combo_list)):
            combo_name = "_".join(combo)
            for i, cov in enumerate(combo):
                covars_key = self.covariate_keys[i]
                if i == 0:
                    emb = self.emb_covars[covars_key][cov]
                else:
                    emb += self.emb_covars[covars_key][cov]
            self.emb_covars_combined[combo_name] = emb

    def get_covars_embeddings_combined(self, return_anndata=True):
        """
        Parameters
        ----------
        return_anndata : bool, optional (default: True)
            Return embedding wrapped into anndata object.

        Returns
        -------
        If return_anndata is True, returns anndata object. Otherwise, doesn't
        return anything. Always saves embeddding in self.emb_covars.
        """
        self._init_covars_embeddings()
        if return_anndata:
            adata = sc.AnnData(np.array(list(self.emb_covars_combined.values())))
            adata.obs["covars"] = self.emb_covars_combined.keys()
            return adata

    def get_covars_embeddings(self, covars_tgt, return_anndata=True):
        """
        Parameters
        ----------
        covars_tgt : str
            Name of covariate for which to return AnnData
        return_anndata : bool, optional (default: True)
            Return embedding wrapped into anndata object.

        Returns
        -------
        If return_anndata is True, returns anndata object. Otherwise, doesn't
        return anything. Always saves embeddding in self.emb_covars.
        """
        self._init_covars_embeddings()

        if return_anndata:
            adata = sc.AnnData(np.array(list(self.emb_covars[covars_tgt].values())))
            adata.obs[covars_tgt] = self.emb_covars[covars_tgt].keys()
            return adata

    def _get_drug_encoding(self, drugs, doses=None):
        """
        Parameters
        ----------
        drugs : str
            Drugs combination as a string, where individual drugs are separated
            with a plus.
        doses : str, optional (default: None)
            Doses corresponding to the drugs combination as a string. Individual
            drugs are separated with a plus.

        Returns
        -------
        One hot encodding for a mixture of drugs.
        """

        drug_mix = np.zeros([1, self.num_drugs])
        atomic_drugs = drugs.split("+")
        doses = str(doses)

        if doses is None:
            doses_list = [1.0] * len(atomic_drugs)
        else:
            doses_list = [float(d) for d in str(doses).split("+")]
        for j, drug in enumerate(atomic_drugs):
            drug_mix += doses_list[j] * self.perts_dict[drug]

        return drug_mix

    def mix_drugs(self, drugs_list, doses_list=None, return_anndata=True):
        """
        Gets a list of drugs combinations to mix, e.g. ['A+B', 'B+C'] and
        corresponding doses.

        Parameters
        ----------
        drugs_list : list
            List of drug combinations, where each drug combination is a string.
            Individual drugs in the combination are separated with a plus.
        doses_list : str, optional (default: None)
            List of corresponding doses, where each dose combination is a string.
            Individual doses in the combination are separated with a plus.
        return_anndata : bool, optional (default: True)
            Return embedding wrapped into anndata object.

        Returns
        -------
        If return_anndata is True, returns anndata structure of the combinations,
        otherwise returns a np.array of corresponding embeddings.
        """

        drug_mix = np.zeros([len(drugs_list), self.num_drugs])
        for i, drug_combo in enumerate(drugs_list):
            drug_mix[i] = self._get_drug_encoding(drug_combo, doses=doses_list[i])

        emb = (
            self.model.compute_drug_embeddings_(
                torch.Tensor(drug_mix).to(self.model.device)
            )
            .cpu()
            .clone()
            .detach()
            .numpy()
        )

        if return_anndata:
            adata = sc.AnnData(emb)
            adata.obs[self.perturbation_key] = drugs_list
            adata.obs[self.dose_key] = doses_list
            return adata
        else:
            return emb

    def latent_dose_response(
        self, perturbations=None, dose=None, contvar_min=0, contvar_max=1, n_points=100
    ):
        """
        Parameters
        ----------
        perturbations : list
            List containing two names for which to return complete pairwise
            dose-response.
        doses : np.array (default: None)
            Doses values. If None, default values will be generated on a grid:
            n_points in range [contvar_min, contvar_max].
        contvar_min : float (default: 0)
            Minimum dose value to generate for default option.
        contvar_max : float (default: 0)
            Maximum dose value to generate for default option.
        n_points : int (default: 100)
            Number of dose points to generate for default option.
        Returns
        -------
        pd.DataFrame
        """
        # dosers work only for atomic drugs. TODO add drug combinations
        self.model.eval()

        if perturbations is None:
            perturbations = self.unique_perts

        if dose is None:
            dose = np.linspace(contvar_min, contvar_max, n_points)
        n_points = len(dose)

        df = pd.DataFrame(columns=[self.perturbation_key, self.dose_key, "response"])
        for drug in perturbations:
            d = np.where(self.perts_dict[drug] == 1)[0][0]
            this_drug = torch.Tensor(dose).to(self.model.device).view(-1, 1)
            if self.model.doser_type == "mlp":
                response = (
                    (self.model.dosers[d](this_drug).sigmoid() * this_drug.gt(0))
                    .cpu()
                    .clone()
                    .detach()
                    .numpy()
                    .reshape(-1)
                )
            else:
                response = (
                    self.model.dosers.one_drug(this_drug.view(-1), d)
                    .cpu()
                    .clone()
                    .detach()
                    .numpy()
                    .reshape(-1)
                )

            df_drug = pd.DataFrame(
                list(zip([drug] * n_points, dose, list(response))),
                columns=[self.perturbation_key, self.dose_key, "response"],
            )
            df = pd.concat([df, df_drug])

        return df

    def latent_dose_response2D(
        self,
        perturbations,
        dose=None,
        contvar_min=0,
        contvar_max=1,
        n_points=100,
    ):
        """
        Parameters
        ----------
        perturbations : list, optional (default: None)
            List of atomic drugs for which to return latent dose response.
            Currently drug combinations are not supported.
        doses : np.array (default: None)
            Doses values. If None, default values will be generated on a grid:
            n_points in range [contvar_min, contvar_max].
        contvar_min : float (default: 0)
            Minimum dose value to generate for default option.
        contvar_max : float (default: 0)
            Maximum dose value to generate for default option.
        n_points : int (default: 100)
            Number of dose points to generate for default option.
        Returns
        -------
        pd.DataFrame
        """
        # dosers work only for atomic drugs. TODO add drug combinations

        assert len(perturbations) == 2, "You should provide a list of 2 perturbations."

        self.model.eval()

        if dose is None:
            dose = np.linspace(contvar_min, contvar_max, n_points)
        n_points = len(dose)

        df = pd.DataFrame(columns=perturbations + ["response"])
        response = {}

        for drug in perturbations:
            d = np.where(self.perts_dict[drug] == 1)[0][0]
            this_drug = torch.Tensor(dose).to(self.model.device).view(-1, 1)
            if self.model.doser_type == "mlp":
                response[drug] = (
                    (self.model.dosers[d](this_drug).sigmoid() * this_drug.gt(0))
                    .cpu()
                    .clone()
                    .detach()
                    .numpy()
                    .reshape(-1)
                )
            else:
                response[drug] = (
                    self.model.dosers.one_drug(this_drug.view(-1), d)
                    .cpu()
                    .clone()
                    .detach()
                    .numpy()
                    .reshape(-1)
                )

        l = 0
        for i in range(len(dose)):
            for j in range(len(dose)):
                df.loc[l] = [
                    dose[i],
                    dose[j],
                    response[perturbations[0]][i] + response[perturbations[1]][j],
                ]
                l += 1

        return df

    def compute_comb_emb(self, thrh=30):
        """
        Generates an AnnData object containing all the latent vectors of the
        cov+dose*pert combinations seen during training.
        Called in api.compute_uncertainty(), stores the AnnData in self.comb_emb.

        Parameters
        ----------
        Returns
        -------
        """
        if self.seen_covars_perts["training"] is None:
            raise ValueError("Need to run parse_training_conditions() first!")

        emb_covars = self.get_covars_embeddings_combined(return_anndata=True)

        # Generate adata with all cov+pert latent vect combinations
        tmp_ad_list = []
        for cov_pert in self.seen_covars_perts["training"]:
            if self.num_measured_points["training"][cov_pert] > thrh:
                *cov_list, pert_loop, dose_loop = cov_pert.split("_")
                cov_loop = "_".join(cov_list)
                emb_perts_loop = []
                if "+" in pert_loop:
                    pert_loop_list = pert_loop.split("+")
                    dose_loop_list = dose_loop.split("+")
                    for _dose in pd.Series(dose_loop_list).unique():
                        tmp_ad = self.get_drug_embeddings(dose=float(_dose))
                        tmp_ad.obs["pert_dose"] = tmp_ad.obs.condition + "_" + _dose
                        emb_perts_loop.append(tmp_ad)

                    emb_perts_loop = emb_perts_loop[0].concatenate(emb_perts_loop[1:])
                    X = emb_covars.X[
                        emb_covars.obs.covars == cov_loop
                    ] + np.expand_dims(
                        emb_perts_loop.X[
                            emb_perts_loop.obs.pert_dose.isin(
                                [
                                    pert_loop_list[i] + "_" + dose_loop_list[i]
                                    for i in range(len(pert_loop_list))
                                ]
                            )
                        ].sum(axis=0),
                        axis=0,
                    )
                    if X.shape[0] > 1:
                        raise ValueError("Error with comb computation")
                else:
                    emb_perts = self.get_drug_embeddings(dose=float(dose_loop))
                    X = (
                        emb_covars.X[emb_covars.obs.covars == cov_loop]
                        + emb_perts.X[emb_perts.obs.condition == pert_loop]
                    )
                tmp_ad = sc.AnnData(X=X)
                tmp_ad.obs["cov_pert"] = "_".join([cov_loop, pert_loop, dose_loop])
            tmp_ad_list.append(tmp_ad)

        self.comb_emb = tmp_ad_list[0].concatenate(tmp_ad_list[1:])

    def compute_uncertainty(self, cov, pert, dose, thrh=30):
        """
        Compute uncertainties for the queried covariate+perturbation combination.
        The distance from the closest condition in the training set is used as a
        proxy for uncertainty.

        Parameters
        ----------
        cov: dict
            Provide a value for each covariate (eg. cell_type) as a dictionaty
            for the queried uncertainty (e.g. cov_dict={'cell_type': 'A549'}).
        pert: string
            Perturbation for the queried uncertainty. In case of combinations the
            format has to be 'pertA+pertB'
        dose: string
            String which contains the dose of the perturbation queried. In case
            of combinations the format has to be 'doseA+doseB'

        Returns
        -------
        min_cos_dist: float
            Minimum cosine distance with the training set.
        min_eucl_dist: float
            Minimum euclidean distance with the training set.
        closest_cond_cos: string
            Closest training condition wrt cosine distances.
        closest_cond_eucl: string
            Closest training condition wrt euclidean distances.
        """

        if self.comb_emb is None:
            self.compute_comb_emb(thrh=30)

        # covar_ohe = torch.Tensor(
        #         self.covars_dict[cov]
        #     ).to(self.model.device)

        drug_ohe = torch.Tensor(self._get_drug_encoding(pert, doses=dose)).to(
            self.model.device
        )

        # cov = covar_ohe.expand([1, self.covars_ohe.shape[1]])
        pert = drug_ohe.expand([1, self.drug_ohe.shape[1]])

        drug_emb = self.model.compute_drug_embeddings_(pert).detach().cpu().numpy()

        cond_emb = drug_emb
        for cov_key in cov:
            cond_emb += self.emb_covars[cov_key][cov[cov_key]]
            # self.model.cell_type_embeddings(cov.argmax(1)).detach().cpu().numpy()

        cos_dist = cosine_distances(cond_emb, self.comb_emb.X)[0]
        min_cos_dist = np.min(cos_dist)
        cos_idx = np.argmin(cos_dist)
        closest_cond_cos = self.comb_emb.obs.cov_pert[cos_idx]

        eucl_dist = euclidean_distances(cond_emb, self.comb_emb.X)[0]
        min_eucl_dist = np.min(eucl_dist)
        eucl_idx = np.argmin(eucl_dist)
        closest_cond_eucl = self.comb_emb.obs.cov_pert[eucl_idx]

        return min_cos_dist, min_eucl_dist, closest_cond_cos, closest_cond_eucl

    def predict(
        self,
        genes,
        cov,
        pert,
        dose,
        uncertainty=True,
        return_anndata=True,
        sample=False,
        n_samples=1,
    ):
        """Predict values of control 'genes' conditions specified in df.

        Parameters
        ----------
        genes : np.array
            Control cells.
        cov: dict of lists
            Provide a value for each covariate (eg. cell_type) as a dictionaty
            for the queried uncertainty (e.g. cov_dict={'cell_type': 'A549'}).
        pert: list
            Perturbation for the queried uncertainty. In case of combinations the
            format has to be 'pertA+pertB'
        dose: list
            String which contains the dose of the perturbation queried. In case
            of combinations the format has to be 'doseA+doseB'

        uncertainty: bool (default: True)
            Compute uncertainties for the generated cells.
        return_anndata : bool, optional (default: True)
            Return embedding wrapped into anndata object.
        sample : bool (default: False)
            If sample is True, returns samples from gausssian distribution with
            mean and variance estimated by the model. Otherwise, returns just
            means and variances estimated by the model.
        n_samples : int (default: 10)
            Number of samples to sample if sampling is True.
        Returns
        -------
        If return_anndata is True, returns anndata structure. Otherwise, returns
        np.arrays for gene_means, gene_vars and a data frame for the corresponding
        conditions df_obs.

        """

        assert len(dose) == len(pert), "Check the length of pert, dose"
        for cov_key in cov:
            assert len(cov[cov_key]) == len(pert), "Check the length of covariates"

        df = pd.concat(
            [
                pd.DataFrame({self.perturbation_key: pert, self.dose_key: dose}),
                pd.DataFrame(cov),
            ],
            axis=1,
        )

        self.model.eval()
        num = genes.shape[0]
        dim = genes.shape[1]
        genes = torch.Tensor(genes).to(self.model.device)
        if sample:
            print(
                "Careful! These are sampled values! Better use means and \
                variances for dowstream tasks!"
            )

        gene_means_list = []
        gene_vars_list = []
        df_list = []

        for i in range(len(df)):
            comb_name = pert[i]
            dose_name = dose[i]
            covar_name = {}
            for cov_key in cov:
                covar_name[cov_key] = cov[cov_key][i]

            drug_ohe = torch.Tensor(
                self._get_drug_encoding(comb_name, doses=dose_name)
            ).to(self.model.device)

            drugs = drug_ohe.expand([num, self.drug_ohe.shape[1]])

            covars = []
            for cov_key in self.covariate_keys:
                covar_ohe = torch.Tensor(
                    self.covars_dict[cov_key][covar_name[cov_key]]
                ).to(self.model.device)
                covars.append(covar_ohe.expand([num, covar_ohe.shape[0]]).clone())

            gene_reconstructions = (
                self.model.predict(genes, drugs, covars).cpu().detach().numpy()
            )

            if sample:
                df_list.append(
                    pd.DataFrame(
                        [df.loc[i].values] * num * n_samples, columns=df.columns
                    )
                )
                if self.args['loss_ae'] == 'gauss':
                    dist = Normal(
                        torch.Tensor(gene_reconstructions[:, :dim]),
                        torch.Tensor(gene_reconstructions[:, dim:]),
                    )
                elif self.args['loss_ae'] == 'nb':
                    counts, logits = _convert_mean_disp_to_counts_logits(
                        torch.clamp(
                            torch.Tensor(gene_reconstructions[:, :dim]),
                            min=1e-8,
                            max=1e8,
                        ),
                        torch.clamp(
                            torch.Tensor(gene_reconstructions[:, dim:]),
                            min=1e-8,
                            max=1e8,
                        )
                    )
                    dist = NegativeBinomial(
                        total_count=counts,
                        logits=logits
                    )
                sampled_gexp = (
                    dist.sample(torch.Size([n_samples]))
                    .cpu()
                    .detach()
                    .numpy()
                    .reshape(-1, dim)
                )
                sampled_gexp[sampled_gexp < 0] = 0 #set negative values to 0, since gexp can't be negative
                gene_means_list.append(sampled_gexp)
            else:
                df_list.append(
                    pd.DataFrame([df.loc[i].values] * num, columns=df.columns)
                )

                gene_means_list.append(gene_reconstructions[:, :dim])

            if uncertainty:
                (
                    cos_dist,
                    eucl_dist,
                    closest_cond_cos,
                    closest_cond_eucl,
                ) = self.compute_uncertainty(
                    cov=covar_name, pert=comb_name, dose=dose_name
                )
                df_list[-1] = df_list[-1].assign(
                    uncertainty_cosine=cos_dist,
                    uncertainty_euclidean=eucl_dist,
                    closest_cond_cosine=closest_cond_cos,
                    closest_cond_euclidean=closest_cond_eucl,
                )

            gene_vars_list.append(gene_reconstructions[:, dim:])

        gene_means = np.concatenate(gene_means_list)
        gene_vars = np.concatenate(gene_vars_list)
        df_obs = pd.concat(df_list)
        del df_list, gene_means_list, gene_vars_list

        if return_anndata:
            adata = sc.AnnData(gene_means)
            adata.var_names = self.var_names
            adata.obs = df_obs
            if not sample:
                adata.layers["variance"] = gene_vars

            adata.obs.index = adata.obs.index.astype(str)  # type fix
            del gene_means, gene_vars, df_obs
            return adata
        else:
            return gene_means, gene_vars, df_obs

    def get_latent(
        self,
        genes,
        cov,
        pert,
        dose,
        return_anndata=True,
    ):
        """Get latent values of control 'genes' with conditions specified in df.

        Parameters
        ----------
        genes : np.array
            Control cells.
        cov: dict of lists
            Provide a value for each covariate (eg. cell_type) as a dictionaty
            for the queried uncertainty (e.g. cov_dict={'cell_type': 'A549'}).
        pert: list
            Perturbation for the queried uncertainty. In case of combinations the
            format has to be 'pertA+pertB'
        dose: list
            String which contains the dose of the perturbation queried. In case
            of combinations the format has to be 'doseA+doseB'
        return_anndata : bool, optional (default: True)
            Return embedding wrapped into anndata object.

        Returns
        -------
        If return_anndata is True, returns anndata structure. Otherwise, returns
        np.arrays for latent and a data frame for the corresponding
        conditions df_obs.

        """

        assert len(dose) == len(pert), "Check the length of pert, dose"
        for cov_key in cov:
            assert len(cov[cov_key]) == len(pert), "Check the length of covariates"

        df = pd.concat(
            [
                pd.DataFrame({self.perturbation_key: pert, self.dose_key: dose}),
                pd.DataFrame(cov),
            ],
            axis=1,
        )

        self.model.eval()
        num = genes.shape[0]
        genes = torch.Tensor(genes).to(self.model.device)

        latent_list = []
        df_list = []

        for i in range(len(df)):
            comb_name = pert[i]
            dose_name = dose[i]
            covar_name = {}
            for cov_key in cov:
                covar_name[cov_key] = cov[cov_key][i]

            drug_ohe = torch.Tensor(
                self._get_drug_encoding(comb_name, doses=dose_name)
            ).to(self.model.device)

            drugs = drug_ohe.expand([num, self.drug_ohe.shape[1]])

            covars = []
            for cov_key in self.covariate_keys:
                covar_ohe = torch.Tensor(
                    self.covars_dict[cov_key][covar_name[cov_key]]
                ).to(self.model.device)
                covars.append(covar_ohe.expand([num, covar_ohe.shape[0]]).clone())

            _, latent_treated = self.model.predict(
                    genes,
                    drugs, 
                    covars,
                    return_latent_treated=True,
            )

            latent_treated = latent_treated.cpu().clone().detach().numpy()

            df_list.append(
                pd.DataFrame([df.loc[i].values] * num, columns=df.columns)
            )

            latent_list.append(latent_treated)

        latent = np.concatenate(latent_list)
        df_obs = pd.concat(df_list)
        del df_list

        if return_anndata:
            adata = sc.AnnData(latent)
            adata.obs = df_obs
            adata.obs.index = adata.obs.index.astype(str)  # type fix
            return adata
        else:
            return latent, df_obs

    def get_response(
        self,
        genes_control=None,
        doses=None,
        contvar_min=None,
        contvar_max=None,
        n_points=10,
        ncells_max=100,
        perturbations=None,
        control_name="test_control",
    ):
        """Decoded dose response data frame.

        Parameters
        ----------
        genes_control : np.array (deafult: None)
            Genes for which to predict values. If None, take from 'test_control'
            split in datasets.
        doses : np.array (default: None)
            Doses values. If None, default values will be generated on a grid:
            n_points in range [contvar_min, contvar_max].
        contvar_min : float (default: 0)
            Minimum dose value to generate for default option.
        contvar_max : float (default: 0)
            Maximum dose value to generate for default option.
        n_points : int (default: 100)
            Number of dose points to generate for default option.
        perturbations : list (default: None)
            List of perturbations for dose response

        Returns
        -------
        pd.DataFrame
            of decoded response values of genes and average response.
        """

        if genes_control is None:
            genes_control = self.datasets["test"].subset_condition(control=True).genes

        if contvar_min is None:
            contvar_min = 0
        if contvar_max is None:
            contvar_max = self.max_dose

        self.model.eval()
        # doses = torch.Tensor(np.linspace(contvar_min, contvar_max, n_points))
        if doses is None:
            doses = np.linspace(contvar_min, contvar_max, n_points)

        if perturbations is None:
            perturbations = self.unique_perts

        response = pd.DataFrame(
            columns=self.covariate_keys
            + [self.perturbation_key, self.dose_key, "response"]
            + list(self.var_names)
        )

        if ncells_max < len(genes_control):
            ncells_max = min(ncells_max, len(genes_control))
            idx = torch.LongTensor(
                np.random.choice(range(len(genes_control)), ncells_max, replace=False)
            )
            genes_control = genes_control[idx]

        i = 0
        for covar_combo in self.emb_covars_combined:
            cov_dict = {}
            for i, cov_val in enumerate(covar_combo.split("_")):
                cov_dict[self.covariate_keys[i]] = [cov_val]

            for idr, drug in enumerate(perturbations):
                if not (drug in self.datasets[control_name].ctrl_name):
                    for dose in doses:
                        # TODO handle covars

                        gene_means, _, _ = self.predict(
                            genes_control,
                            cov=cov_dict,
                            pert=[drug],
                            dose=[dose],
                            return_anndata=False,
                        )
                        predicted_data = np.mean(gene_means, axis=0).reshape(-1)

                        response.loc[i] = (
                            covar_combo.split("_")
                            + [drug, dose, np.linalg.norm(predicted_data)]
                            + list(predicted_data)
                        )
                        i += 1
        return response

    def get_response_reference(self, perturbations=None):

        """Computes reference values of the response.

        Parameters
        ----------
        dataset : CompPertDataset
            The file location of the spreadsheet
        perturbations : list (default: None)
            List of perturbations for dose response

        Returns
        -------
        pd.DataFrame
            of decoded response values of genes and average response.
        """
        if perturbations is None:
            perturbations = self.unique_perts

        reference_response_curve = pd.DataFrame(
            columns=self.covariate_keys
            + [self.perturbation_key, self.dose_key, "split", "num_cells", "response"]
            + list(self.var_names)
        )

        dataset_ctr = self.datasets["training"].subset_condition(control=True)

        i = 0
        for split in ["training_treated", "ood"]:
            if split == 'ood':
                dataset = self.datasets[split]
            else:
                dataset = self.datasets["training"].subset_condition(control=False)
            for pert in self.seen_covars_perts[split]:
                *covars, drug, dose_val = pert.split("_")
                if drug in perturbations:
                    if not ("+" in dose_val):
                        dose = float(dose_val)
                    else:
                        dose = dose_val

                    idx = np.where((dataset.pert_categories == pert))[0]

                    if len(idx):
                        y_true = dataset.genes[idx, :].numpy().mean(axis=0)
                        reference_response_curve.loc[i] = (
                            covars
                            + [drug, dose, split, len(idx), np.linalg.norm(y_true)]
                            + list(y_true)
                        )

                        i += 1

        reference_response_curve = reference_response_curve.replace(
            "training_treated", "train"
        )
        return reference_response_curve

    def get_response2D(
        self,
        perturbations,
        covar,
        genes_control=None,
        doses=None,
        contvar_min=None,
        contvar_max=None,
        n_points=10,
        ncells_max=100,
        fixed_drugs="",
        fixed_doses="",
    ):
        """Decoded dose response data frame.

        Parameters
        ----------
        perturbations : list
            List of length 2 of perturbations for dose response.
        covar : dict
            Name of a covariate for which to compute dose-response.
        genes_control : np.array (deafult: None)
            Genes for which to predict values. If None, take from 'test_control'
            split in datasets.
        doses : np.array (default: None)
            Doses values. If None, default values will be generated on a grid:
            n_points in range [contvar_min, contvar_max].
        contvar_min : float (default: 0)
            Minimum dose value to generate for default option.
        contvar_max : float (default: 0)
            Maximum dose value to generate for default option.
        n_points : int (default: 100)
            Number of dose points to generate for default option.

        Returns
        -------
        pd.DataFrame
            of decoded response values of genes and average response.
        """

        assert len(perturbations) == 2, "You should provide a list of 2 perturbations."

        if contvar_min is None:
            contvar_min = self.min_dose

        if contvar_max is None:
            contvar_max = self.max_dose

        self.model.eval()
        # doses = torch.Tensor(np.linspace(contvar_min, contvar_max, n_points))
        if doses is None:
            doses = np.linspace(contvar_min, contvar_max, n_points)

        # genes_control = dataset.genes[dataset.indices['control']]
        if genes_control is None:
            genes_control = self.datasets["test"].subset_condition(control=True).genes

        ncells_max = min(ncells_max, len(genes_control))
        idx = torch.LongTensor(np.random.choice(range(len(genes_control)), ncells_max))
        genes_control = genes_control[idx]

        response = pd.DataFrame(
            columns=perturbations + ["response"] + list(self.var_names)
        )

        drug = perturbations[0] + "+" + perturbations[1]

        dose_vals = [f"{d[0]}+{d[1]}" for d in itertools.product(*[doses, doses])]
        dose_comb = [list(d) for d in itertools.product(*[doses, doses])]

        i = 0
        if not (drug in ["Vehicle", "EGF", "unst", "control", "ctrl"]):
            for dose in dose_vals:
                gene_means, _, _ = self.predict(
                    genes_control,
                    cov=covar,
                    pert=[drug + fixed_drugs],
                    dose=[dose + fixed_doses],
                    return_anndata=False,
                )

                predicted_data = np.mean(gene_means, axis=0).reshape(-1)

                response.loc[i] = (
                    dose_comb[i]
                    + [np.linalg.norm(predicted_data)]
                    + list(predicted_data)
                )
                i += 1

        return response

    def get_cycle_uncertainty(
        self, genes_from, df_from, df_to, ncells_max=100, direction="forward"
    ):

        """Uncertainty for a single condition.

        Parameters
        ----------
        genes_from: torch.Tensor
            Genes for comparison.
        df_from: pd.DataFrame
            Full description of the condition.
        df_to: pd.DataFrame
            Full description of the control condition.
        ncells_max: int, optional (defaul: 100)
            Max number of cells to use.
        Returns
        -------
        tuple
            with uncertainty estimations: (MSE, 1-R2).
        """
        self.model.eval()
        genes_control = genes_from.clone().detach()

        if ncells_max < len(genes_control):
            idx = torch.LongTensor(
                np.random.choice(range(len(genes_control)), ncells_max, replace=False)
            )
            genes_control = genes_control[idx]

        gene_condition, _, _ = self.predict(
            genes_control, df_to, return_anndata=False, sample=False
        )
        gene_condition = torch.Tensor(gene_condition).clone().detach()
        gene_return, _, _ = self.predict(
            gene_condition, df_from, return_anndata=False, sample=False
        )

        if direction == "forward":
            # control -> condition -> control'
            genes_control = genes_control.numpy()
            ctr = np.mean(genes_control, axis=0)
            ret = np.mean(gene_return, axis=0)
            return np.mean((genes_control - gene_return) ** 2), 1 - r2_score(ctr, ret)
        else:
            # control -> condition -> control' -> condition'
            gene_return = torch.Tensor(gene_return).clone().detach()
            gene_condition_return, _, _ = self.predict(
                gene_return, df_to, return_anndata=False, sample=False
            )
            gene_condition = gene_condition.numpy()
            ctr = np.mean(gene_condition, axis=0)
            ret = np.mean(gene_condition_return, axis=0)
            return np.mean((gene_condition - gene_condition_return) ** 2), 1 - r2_score(
                ctr, ret
            )

    def print_complete_cycle_uncertainty(
        self,
        datasets,
        datasets_ctr,
        ncells_max=1000,
        split_list=["test", "ood"],
        direction="forward",
    ):
        uncert = pd.DataFrame(
            columns=[
                self.covars_key,
                self.perturbation_key,
                self.dose_key,
                "split",
                "MSE",
                "1-R2",
            ]
        )

        ctr_covar, ctrl_name, ctr_dose = datasets_ctr.pert_categories[0].split("_")
        df_ctrl = pd.DataFrame(
            {
                self.perturbation_key: [ctrl_name],
                self.dose_key: [ctr_dose],
                self.covars_key: [ctr_covar],
            }
        )

        i = 0
        for split in split_list:
            dataset = datasets[split]
            print(split)
            for pert_cat in np.unique(dataset.pert_categories):
                idx = np.where(dataset.pert_categories == pert_cat)[0]
                genes = dataset.genes[idx, :]

                covar, pert, dose = pert_cat.split("_")
                df_cond = pd.DataFrame(
                    {
                        self.perturbation_key: [pert],
                        self.dose_key: [dose],
                        self.covars_key: [covar],
                    }
                )

                if direction == "back":
                    # condition -> control -> condition
                    uncert.loc[i] = [covar, pert, dose, split] + list(
                        self.get_cycle_uncertainty(
                            genes, df_cond, df_ctrl, ncells_max=ncells_max
                        )
                    )
                else:
                    # control -> condition -> control
                    uncert.loc[i] = [covar, pert, dose, split] + list(
                        self.get_cycle_uncertainty(
                            datasets_ctr.genes,
                            df_ctrl,
                            df_cond,
                            ncells_max=ncells_max,
                            direction=direction,
                        )
                    )

                i += 1

        return uncert

    def evaluate_r2(self, dataset, genes_control):
        """
        Measures different quality metrics about an CPA `autoencoder`, when
        tasked to translate some `genes_control` into each of the drug/cell_type
        combinations described in `dataset`.

        Considered metrics are R2 score about means and variances for all genes, as
        well as R2 score about means and variances about differentially expressed
        (_de) genes.
        """
        self.model.eval()
        scores = pd.DataFrame(
            columns=self.covariate_keys
            + [
                self.perturbation_key,
                self.dose_key,
                "R2_mean",
                "R2_mean_DE",
                "R2_var",
                "R2_var_DE",
                "num_cells",
            ]
        )

        num, dim = genes_control.size(0), genes_control.size(1)

        total_cells = len(dataset)

        icond = 0
        for pert_category in np.unique(dataset.pert_categories):
            # pert_category category contains: 'celltype_perturbation_dose' info
            de_idx = np.where(
                dataset.var_names.isin(np.array(dataset.de_genes[pert_category]))
            )[0]

            idx = np.where(dataset.pert_categories == pert_category)[0]
            *covars, pert, dose = pert_category.split("_")
            cov_dict = {}
            for i, cov_key in enumerate(self.covariate_keys):
                cov_dict[cov_key] = [covars[i]]

            if len(idx) > 0:
                mean_predict, var_predict, _ = self.predict(
                    genes_control,
                    cov=cov_dict,
                    pert=[pert],
                    dose=[dose],
                    return_anndata=False,
                    sample=False,
                )

                # estimate metrics only for reasonably-sized drug/cell-type combos
                y_true = dataset.genes[idx, :].numpy()

                # true means and variances
                yt_m = y_true.mean(axis=0)
                yt_v = y_true.var(axis=0)
                # predicted means and variances
                yp_m = mean_predict.mean(0)
                yp_v = var_predict.mean(0)

                mean_score = r2_score(yt_m, yp_m)
                var_score = r2_score(yt_v, yp_v)

                mean_score_de = r2_score(yt_m[de_idx], yp_m[de_idx])
                var_score_de = r2_score(yt_v[de_idx], yp_v[de_idx])

                scores.loc[icond] = pert_category.split("_") + [
                    mean_score,
                    mean_score_de,
                    var_score,
                    var_score_de,
                    len(idx),
                ]
                icond += 1

        return scores


def get_reference_from_combo(perturbations_list, datasets, splits=["training", "ood"]):
    """
    A simple function that produces a pd.DataFrame of individual
    drugs-doses combinations used among the splits (for a fixed covariate).
    """
    df_list = []
    for split_name in splits:
        full_dataset = datasets[split_name]
        ref = {"num_cells": []}
        for pp in perturbations_list:
            ref[pp] = []

        ndrugs = len(perturbations_list)
        for pert_cat in np.unique(full_dataset.pert_categories):
            _, pert, dose = pert_cat.split("_")
            pert_list = pert.split("+")
            if set(pert_list) == set(perturbations_list):
                dose_list = dose.split("+")
                ncells = len(
                    full_dataset.pert_categories[
                        full_dataset.pert_categories == pert_cat
                    ]
                )
                for j in range(ndrugs):
                    ref[pert_list[j]].append(float(dose_list[j]))
                ref["num_cells"].append(ncells)
                print(pert, dose, ncells)
        df = pd.DataFrame.from_dict(ref)
        df["split"] = split_name
        df_list.append(df)

    return pd.concat(df_list)


def linear_interp(y1, y2, x1, x2, x):
    a = (y1 - y2) / (x1 - x2)
    b = y1 - a * x1
    y = a * x + b
    return y


def evaluate_r2_benchmark(cpa_api, datasets, pert_category, pert_category_list):
    scores = pd.DataFrame(
        columns=[
            cpa_api.covars_key,
            cpa_api.perturbation_key,
            cpa_api.dose_key,
            "R2_mean",
            "R2_mean_DE",
            "R2_var",
            "R2_var_DE",
            "num_cells",
            "benchmark",
            "method",
        ]
    )

    de_idx = np.where(
        datasets["ood"].var_names.isin(
            np.array(datasets["ood"].de_genes[pert_category])
        )
    )[0]
    idx = np.where(datasets["ood"].pert_categories == pert_category)[0]
    y_true = datasets["ood"].genes[idx, :].numpy()
    # true means and variances
    yt_m = y_true.mean(axis=0)
    yt_v = y_true.var(axis=0)

    icond = 0
    if len(idx) > 0:
        for pert_category_predict in pert_category_list:
            if "+" in pert_category_predict:
                pert1, pert2 = pert_category_predict.split("+")
                idx_pred1 = np.where(datasets["training"].pert_categories == pert1)[0]
                idx_pred2 = np.where(datasets["training"].pert_categories == pert2)[0]

                y_pred1 = datasets["training"].genes[idx_pred1, :].numpy()
                y_pred2 = datasets["training"].genes[idx_pred2, :].numpy()

                x1 = float(pert1.split("_")[2])
                x2 = float(pert2.split("_")[2])
                x = float(pert_category.split("_")[2])
                yp_m1 = y_pred1.mean(axis=0)
                yp_m2 = y_pred2.mean(axis=0)
                yp_v1 = y_pred1.var(axis=0)
                yp_v2 = y_pred2.var(axis=0)

                yp_m = linear_interp(yp_m1, yp_m2, x1, x2, x)
                yp_v = linear_interp(yp_v1, yp_v2, x1, x2, x)

            #                     yp_m = (y_pred1.mean(axis=0) + y_pred2.mean(axis=0))/2
            #                     yp_v = (y_pred1.var(axis=0) + y_pred2.var(axis=0))/2

            else:
                idx_pred = np.where(
                    datasets["training"].pert_categories == pert_category_predict
                )[0]
                print(pert_category_predict, len(idx_pred))
                y_pred = datasets["training"].genes[idx_pred, :].numpy()
                # predicted means and variances
                yp_m = y_pred.mean(axis=0)
                yp_v = y_pred.var(axis=0)

            mean_score = r2_score(yt_m, yp_m)
            var_score = r2_score(yt_v, yp_v)

            mean_score_de = r2_score(yt_m[de_idx], yp_m[de_idx])
            var_score_de = r2_score(yt_v[de_idx], yp_v[de_idx])
            scores.loc[icond] = pert_category.split("_") + [
                mean_score,
                mean_score_de,
                var_score,
                var_score_de,
                len(idx),
                pert_category_predict,
                "benchmark",
            ]
            icond += 1

    return scores
