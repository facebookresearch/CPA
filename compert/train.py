# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import json
import argparse

import torch
import numpy as np
from collections import defaultdict

from compert.data import load_dataset_splits
from compert.model import ComPert

from sklearn.metrics import r2_score, balanced_accuracy_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import time

def pjson(s):
    """
    Prints a string in JSON format and flushes stdout
    """
    print(json.dumps(s), flush=True)


def evaluate_disentanglement(autoencoder, dataset, nonlinear=False):
    """
    Given a ComPert model, this function measures the correlation between
    its latent space and 1) a dataset's drug vectors 2) a datasets covariate
    vectors.

    """
    _, latent_basal = autoencoder.predict(
        dataset.genes,
        dataset.drugs,
        dataset.cell_types,
        return_latent_basal=True)

    latent_basal = latent_basal.detach().cpu().numpy()

    if nonlinear:
        clf = KNeighborsClassifier(
            n_neighbors=int(np.sqrt(len(latent_basal))))
    else:
        clf = LogisticRegression(solver="liblinear",
                                 multi_class="auto",
                                 max_iter=10000)

    pert_scores = cross_val_score(
        clf,
        StandardScaler().fit_transform(latent_basal), dataset.drugs_names,
        scoring=make_scorer(balanced_accuracy_score), cv=5, n_jobs=-1)

    if len(np.unique(dataset.cell_types_names)) > 1:
        cov_scores = cross_val_score(
            clf,
            StandardScaler().fit_transform(latent_basal), dataset.cell_types_names,
            scoring=make_scorer(balanced_accuracy_score), cv=5, n_jobs=-1)
        return np.mean(pert_scores), np.mean(cov_scores)
    else:
        return np.mean(pert_scores), 0



def evaluate_r2(autoencoder, dataset, genes_control):
    """
    Measures different quality metrics about an ComPert `autoencoder`, when
    tasked to translate some `genes_control` into each of the drug/cell_type
    combinations described in `dataset`.

    Considered metrics are R2 score about means and variances for all genes, as
    well as R2 score about means and variances about differentially expressed
    (_de) genes.
    """

    mean_score, var_score, mean_score_de, var_score_de = [], [], [], []
    num, dim = genes_control.size(0), genes_control.size(1)

    total_cells = len(dataset)

    for pert_category in np.unique(dataset.pert_categories):
        # pert_category category contains: 'celltype_perturbation_dose' info
        de_idx = np.where(
            dataset.var_names.isin(
                np.array(dataset.de_genes[pert_category])))[0]

        idx = np.where(dataset.pert_categories == pert_category)[0]

        if len(idx) > 30:
            emb_drugs = dataset.drugs[idx][0].view(
                1, -1).repeat(num, 1).clone()
            emb_cts = dataset.cell_types[idx][0].view(
                1, -1).repeat(num, 1).clone()

            genes_predict = autoencoder.predict(
                genes_control, emb_drugs, emb_cts).detach().cpu()

            mean_predict = genes_predict[:, :dim]
            var_predict = genes_predict[:, dim:]

            # estimate metrics only for reasonably-sized drug/cell-type combos

            y_true = dataset.genes[idx, :].numpy()

            # true means and variances
            yt_m = y_true.mean(axis=0)
            yt_v = y_true.var(axis=0)
            # predicted means and variances
            yp_m = mean_predict.mean(0)
            yp_v = var_predict.mean(0)

            mean_score.append(r2_score(yt_m, yp_m))
            var_score.append(r2_score(yt_v, yp_v))

            mean_score_de.append(r2_score(yt_m[de_idx], yp_m[de_idx]))
            var_score_de.append(r2_score(yt_v[de_idx], yp_v[de_idx]))

    return [np.mean(s) if len(s) else -1
            for s in [mean_score, mean_score_de, var_score, var_score_de]]


def evaluate(autoencoder, datasets):
    """
    Measure quality metrics using `evaluate()` on the training, test, and
    out-of-distributiion (ood) splits.
    """

    autoencoder.eval()
    with torch.no_grad():
        stats_test = evaluate_r2(
            autoencoder,
            datasets["test_treated"],
            datasets["test_control"].genes)

        stats_disent_pert, stats_disent_cov = evaluate_disentanglement(
            autoencoder, datasets["test"])

        evaluation_stats = {
            "training": evaluate_r2(
                autoencoder,
                datasets["training_treated"],
                datasets["training_control"].genes),
            "test": stats_test,
            "ood": evaluate_r2(
                autoencoder,
                datasets["ood"],
                datasets["test_control"].genes),
            "perturbation disentanglement": stats_disent_pert,
            "optimal for perturbations": 1/datasets['test'].num_drugs,
            "covariate disentanglement": stats_disent_cov,
            "optimal for covariates": 1/datasets['test'].num_cell_types,
        }
    autoencoder.train()
    return evaluation_stats


def prepare_compert(args, state_dict=None):
    """
    Instantiates autoencoder and dataset to run an experiment.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    datasets = load_dataset_splits(
        args["dataset_path"],
        args["perturbation_key"],
        args["dose_key"],
        args["cell_type_key"],
        args["split_key"])

    autoencoder = ComPert(
        datasets["training"].num_genes,
        datasets["training"].num_drugs,
        datasets["training"].num_cell_types,
        device=device,
        seed=args["seed"],
        loss_ae=args["loss_ae"],
        doser_type=args["doser_type"],
        patience=args["patience"],
        hparams=args["hparams"],
        decoder_activation=args["decoder_activation"],
    )
    if state_dict is not None:
        autoencoder.load_state_dict(state_dict)

    return autoencoder, datasets


def train_compert(args, return_model=False):
    """
    Trains a ComPert autoencoder
    """

    autoencoder, datasets = prepare_compert(args)

    datasets.update({
        "loader_tr": torch.utils.data.DataLoader(
                        datasets["training"],
                        batch_size=autoencoder.hparams["batch_size"],
                        shuffle=True)
    })

    pjson({"training_args": args})
    pjson({"autoencoder_params": autoencoder.hparams})

    start_time = time.time()
    for epoch in range(args["max_epochs"]):
        epoch_training_stats = defaultdict(float)

        for genes, drugs, cell_types in datasets["loader_tr"]:
            minibatch_training_stats = autoencoder.update(
                genes, drugs, cell_types)

            for key, val in minibatch_training_stats.items():
                epoch_training_stats[key] += val

        for key, val in epoch_training_stats.items():
            epoch_training_stats[key] = val / len(datasets["loader_tr"])
            if not (key in autoencoder.history.keys()):
                autoencoder.history[key] = []
            autoencoder.history[key].append(val)
        autoencoder.history['epoch'].append(epoch)

        ellapsed_minutes = (time.time() - start_time) / 60
        autoencoder.history['elapsed_time_min'] = ellapsed_minutes

        # decay learning rate if necessary
        # also check stopping condition: patience ran out OR
        # time ran out OR max epochs achieved
        stop = ellapsed_minutes > args["max_minutes"] or \
            (epoch == args["max_epochs"] - 1)

        if (epoch % args["checkpoint_freq"]) == 0 or stop:
            evaluation_stats = evaluate(autoencoder, datasets)
            for key, val in evaluation_stats.items():
                if not (key in autoencoder.history.keys()):
                    autoencoder.history[key] = []
                autoencoder.history[key].append(val)
            autoencoder.history['stats_epoch'].append(epoch)

            pjson({
                "epoch": epoch,
                "training_stats": epoch_training_stats,
                "evaluation_stats": evaluation_stats,
                "ellapsed_minutes": ellapsed_minutes
            })

            torch.save(
                (autoencoder.state_dict(), args, autoencoder.history),
                os.path.join(
                    args["save_dir"],
                    "model_seed={}_epoch={}.pt".format(args["seed"], epoch)))

            pjson({"model_saved": "model_seed={}_epoch={}.pt\n".format(
                args["seed"], epoch)})
            stop = stop or autoencoder.early_stopping(
                np.mean(evaluation_stats["test"]))
            if stop:
                pjson({"early_stop": epoch})
                break

    if return_model:
        return autoencoder, datasets


def parse_arguments():
    """
    Read arguments if this script is called from a terminal.
    """

    parser = argparse.ArgumentParser(description='Drug combinations.')
    # dataset arguments
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--perturbation_key', type=str, default="condition")
    parser.add_argument('--dose_key', type=str, default="dose_val")
    parser.add_argument('--cell_type_key', type=str, default="cell_type")
    parser.add_argument('--split_key', type=str, default="split")
    parser.add_argument('--loss_ae', type=str, default='gauss')
    parser.add_argument('--doser_type', type=str, default='sigm')
    parser.add_argument('--decoder_activation', type=str, default='linear')

    # ComPert arguments (see set_hparams_() in compert.model.ComPert)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hparams', type=str, default="")

    # training arguments
    parser.add_argument('--max_epochs', type=int, default=2000)
    parser.add_argument('--max_minutes', type=int, default=300)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--checkpoint_freq', type=int, default=20)

    # output folder
    parser.add_argument('--save_dir', type=str, required=True)
    # number of trials when executing compert.sweep
    parser.add_argument('--sweep_seeds', type=int, default=200)
    return dict(vars(parser.parse_args()))


if __name__ == "__main__":
    train_compert(parse_arguments())
