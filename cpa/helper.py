# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import warnings

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import r2_score
from scipy.sparse import issparse
from scipy.stats import wasserstein_distance
import torch

warnings.filterwarnings("ignore")

import sys

if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.simplefilter(action="ignore", category=FutureWarning)

def _convert_mean_disp_to_counts_logits(mu, theta, eps=1e-6):
    r"""NB parameterizations conversion
    Parameters
    ----------
    mu :
        mean of the NB distribution.
    theta :
        inverse overdispersion.
    eps :
        constant used for numerical log stability. (Default value = 1e-6)
    Returns
    -------
    type
        the number of failures until the experiment is stopped
        and the success probability.
    """
    assert (mu is None) == (
        theta is None
    ), "If using the mu/theta NB parameterization, both parameters must be specified"
    logits = (mu + eps).log() - (theta + eps).log()
    total_count = theta
    return total_count, logits


def rank_genes_groups_by_cov(
    adata,
    groupby,
    control_group,
    covariate,
    pool_doses=False,
    n_genes=50,
    rankby_abs=True,
    key_added="rank_genes_groups_cov",
    return_dict=False,
):

    """
    Function that generates a list of differentially expressed genes computed
    separately for each covariate category, and using the respective control
    cells as reference.

    Usage example:

    rank_genes_groups_by_cov(
        adata,
        groupby='cov_product_dose',
        covariate_key='cell_type',
        control_group='Vehicle_0'
    )

    Parameters
    ----------
    adata : AnnData
        AnnData dataset
    groupby : str
        Obs column that defines the groups, should be
        cartesian product of covariate_perturbation_cont_var,
        it is important that this format is followed.
    control_group : str
        String that defines the control group in the groupby obs
    covariate : str
        Obs column that defines the main covariate by which we
        want to separate DEG computation (eg. cell type, species, etc.)
    n_genes : int (default: 50)
        Number of DEGs to include in the lists
    rankby_abs : bool (default: True)
        If True, rank genes by absolute values of the score, thus including
        top downregulated genes in the top N genes. If False, the ranking will
        have only upregulated genes at the top.
    key_added : str (default: 'rank_genes_groups_cov')
        Key used when adding the dictionary to adata.uns
    return_dict : str (default: False)
        Signals whether to return the dictionary or not

    Returns
    -------
    Adds the DEG dictionary to adata.uns

    If return_dict is True returns:
    gene_dict : dict
        Dictionary where groups are stored as keys, and the list of DEGs
        are the corresponding values

    """

    gene_dict = {}
    cov_categories = adata.obs[covariate].unique()
    for cov_cat in cov_categories:
        print(cov_cat)
        # name of the control group in the groupby obs column
        control_group_cov = "_".join([cov_cat, control_group])

        # subset adata to cells belonging to a covariate category
        adata_cov = adata[adata.obs[covariate] == cov_cat]

        # compute DEGs
        sc.tl.rank_genes_groups(
            adata_cov,
            groupby=groupby,
            reference=control_group_cov,
            rankby_abs=rankby_abs,
            n_genes=n_genes,
        )

        # add entries to dictionary of gene sets
        de_genes = pd.DataFrame(adata_cov.uns["rank_genes_groups"]["names"])
        for group in de_genes:
            gene_dict[group] = de_genes[group].tolist()

    adata.uns[key_added] = gene_dict

    if return_dict:
        return gene_dict


def rank_genes_groups(
    adata,
    groupby,
    pool_doses=False,
    n_genes=50,
    rankby_abs=True,
    key_added="rank_genes_groups_cov",
    return_dict=False,
):

    """
    Function that generates a list of differentially expressed genes computed
    separately for each covariate category, and using the respective control
    cells as reference.

    Usage example:

    rank_genes_groups_by_cov(
        adata,
        groupby='cov_product_dose',
        covariate_key='cell_type',
        control_group='Vehicle_0'
    )

    Parameters
    ----------
    adata : AnnData
        AnnData dataset
    groupby : str
        Obs column that defines the groups, should be
        cartesian product of covariate_perturbation_cont_var,
        it is important that this format is followed.
    control_group : str
        String that defines the control group in the groupby obs
    covariate : str
        Obs column that defines the main covariate by which we
        want to separate DEG computation (eg. cell type, species, etc.)
    n_genes : int (default: 50)
        Number of DEGs to include in the lists
    rankby_abs : bool (default: True)
        If True, rank genes by absolute values of the score, thus including
        top downregulated genes in the top N genes. If False, the ranking will
        have only upregulated genes at the top.
    key_added : str (default: 'rank_genes_groups_cov')
        Key used when adding the dictionary to adata.uns
    return_dict : str (default: False)
        Signals whether to return the dictionary or not

    Returns
    -------
    Adds the DEG dictionary to adata.uns

    If return_dict is True returns:
    gene_dict : dict
        Dictionary where groups are stored as keys, and the list of DEGs
        are the corresponding values

    """

    covars_comb = []
    for i in range(len(adata)):
        cov = "_".join(adata.obs["cov_drug_dose_name"].values[i].split("_")[:-2])
        covars_comb.append(cov)
    adata.obs["covars_comb"] = covars_comb

    gene_dict = {}
    for cov_cat in np.unique(adata.obs["covars_comb"].values):
        adata_cov = adata[adata.obs["covars_comb"] == cov_cat]
        control_group_cov = (
            adata_cov[adata_cov.obs["control"] == 1].obs[groupby].values[0]
        )

        # compute DEGs
        sc.tl.rank_genes_groups(
            adata_cov,
            groupby=groupby,
            reference=control_group_cov,
            rankby_abs=rankby_abs,
            n_genes=n_genes,
        )

        # add entries to dictionary of gene sets
        de_genes = pd.DataFrame(adata_cov.uns["rank_genes_groups"]["names"])
        for group in de_genes:
            gene_dict[group] = de_genes[group].tolist()

    adata.uns[key_added] = gene_dict

    if return_dict:
        return gene_dict

# def evaluate_r2_(adata, pred_adata, condition_key, sampled=False):
#     r2_list = []
#     if issparse(adata.X): 
#         adata.X = adata.X.A
#     if issparse(pred_adata.X): 
#         pred_adata.X = pred_adata.X.A
#     for cond in pred_adata.obs[condition_key].unique():
#         adata_ = adata[adata.obs[condition_key] == cond]
#         pred_adata_ = pred_adata[pred_adata.obs[condition_key] == cond]
#         r2_mean = r2_score(adata_.X.mean(0), pred_adata_.X.mean(0))
#         if sampled:
#             r2_var = r2_score(adata_.X.var(0), pred_adata_.X.var(0))
#         else:
#             r2_var = r2_score(
#                 adata_.X.var(0), 
#                 pred_adata_.layers['variance'].var(0)
#             )
#         r2_list.append(
#             {
#                 'condition': cond,
#                 'r2_mean': r2_mean,
#                 'r2_var': r2_var,
#             }
#         )
#     r2_df = pd.DataFrame(r2_list).set_index('condition')
#     return r2_df

def evaluate_r2_(adata, pred_adata, condition_key, sampled=False, de_genes_dict=None):
    r2_list = []
    if issparse(adata.X): 
        adata.X = adata.X.A
    if issparse(pred_adata.X): 
        pred_adata.X = pred_adata.X.A
    for cond in pred_adata.obs[condition_key].unique():
        adata_ = adata[adata.obs[condition_key] == cond]
        pred_adata_ = pred_adata[pred_adata.obs[condition_key] == cond]
        r2_mean = r2_score(adata_.X.mean(0), pred_adata_.X.mean(0))
        if sampled:
            r2_var = r2_score(adata_.X.var(0), pred_adata_.X.var(0))
        else:
            r2_var = r2_score(
                adata_.X.var(0), 
                pred_adata_.layers['variance'].var(0)
            )
        r2_list.append(
            {
                'condition': cond,
                'r2_mean': r2_mean,
                'r2_var': r2_var,
            }
        )
        if de_genes_dict:
            de_genes = de_genes_dict[cond]
            sub_adata_ = adata_[:, de_genes]
            sub_pred_adata_ = pred_adata_[:, de_genes]
            r2_mean_deg = r2_score(sub_adata_.X.mean(0), sub_pred_adata_.X.mean(0))
            if sampled:
                r2_var_deg = r2_score(sub_adata_.X.var(0), sub_pred_adata_.X.var(0))
            else:
                r2_var_deg = r2_score(
                    sub_adata_.X.var(0), 
                    sub_pred_adata_.layers['variance'].var(0)
                )
            r2_list[-1]['r2_mean_deg'] = r2_mean_deg
            r2_list[-1]['r2_var_deg'] = r2_var_deg
    r2_df = pd.DataFrame(r2_list).set_index('condition')
    return r2_df
    
def evaluate_mmd(adata, pred_adata, condition_key, de_genes_dict=None):
    mmd_list = []
    for cond in pred_adata.obs[condition_key].unique():
        adata_ = adata[adata.obs[condition_key] == cond].copy()
        pred_adata_ = pred_adata[pred_adata.obs[condition_key] == cond].copy()
        if issparse(adata_.X): 
            adata_.X = adata_.X.A
        if issparse(pred_adata_.X): 
            pred_adata_.X = pred_adata_.X.A

        mmd = mmd_loss_calc(torch.Tensor(adata_.X), torch.Tensor(pred_adata_.X))
        mmd_list.append(
            {
                'condition': cond,
                'mmd': mmd.detach().cpu().numpy()
            }
        )
        if de_genes_dict:
            de_genes = de_genes_dict[cond]
            sub_adata_ = adata_[:, de_genes]
            sub_pred_adata_ = pred_adata_[:, de_genes]
            mmd_deg = mmd_loss_calc(torch.Tensor(sub_adata_.X), torch.Tensor(sub_pred_adata_.X))
            mmd_list[-1]['mmd_deg'] = mmd_deg.detach().cpu().numpy()
    mmd_df = pd.DataFrame(mmd_list).set_index('condition')
    return mmd_df

def evaluate_emd(adata, pred_adata, condition_key, de_genes_dict=None):
    emd_list = []
    for cond in pred_adata.obs[condition_key].unique():
        adata_ = adata[adata.obs[condition_key] == cond].copy()
        pred_adata_ = pred_adata[pred_adata.obs[condition_key] == cond].copy()
        if issparse(adata_.X): 
            adata_.X = adata_.X.A
        if issparse(pred_adata_.X): 
            pred_adata_.X = pred_adata_.X.A
        wd = []
        for i, _ in enumerate(adata_.var_names):
            wd.append(
                wasserstein_distance(torch.Tensor(adata_.X[:, i]), torch.Tensor(pred_adata_.X[:, i]))
            )
        emd_list.append(
            {
                'condition': cond,
                'emd': np.mean(wd)
            }
        )
        if de_genes_dict:
            de_genes = de_genes_dict[cond]
            sub_adata_ = adata_[:, de_genes]
            sub_pred_adata_ = pred_adata_[:, de_genes]
            wd_deg = []
            for i, _ in enumerate(sub_adata_.var_names):
                wd_deg.append(
                    wasserstein_distance(torch.Tensor(sub_adata_.X[:, i]), torch.Tensor(sub_pred_adata_.X[:, i]))
                )
            emd_list[-1]['emd_deg'] = np.mean(wd_deg)
    emd_df = pd.DataFrame(emd_list).set_index('condition')
    return emd_df

def pairwise_distance(x, y):
    x = x.view(x.shape[0], x.shape[1], 1)
    y = torch.transpose(y, 0, 1)
    output = torch.sum((x - y) ** 2, 1)
    output = torch.transpose(output, 0, 1)
    return output


def gaussian_kernel_matrix(x, y, alphas):
    """Computes multiscale-RBF kernel between x and y.
       Parameters
       ----------
       x: torch.Tensor
            Tensor with shape [batch_size, z_dim].
       y: torch.Tensor
            Tensor with shape [batch_size, z_dim].
       alphas: Tensor
       Returns
       -------
       Returns the computed multiscale-RBF kernel between x and y.
    """

    dist = pairwise_distance(x, y).contiguous()
    dist_ = dist.view(1, -1)

    alphas = alphas.view(alphas.shape[0], 1)
    beta = 1. / (2. * alphas)

    s = torch.matmul(beta, dist_)

    return torch.sum(torch.exp(-s), 0).view_as(dist)


def mmd_loss_calc(source_features, target_features):
    """Initializes Maximum Mean Discrepancy(MMD) between source_features and target_features.
       - Gretton, Arthur, et al. "A Kernel Two-Sample Test". 2012.
       Parameters
       ----------
       source_features: torch.Tensor
            Tensor with shape [batch_size, z_dim]
       target_features: torch.Tensor
            Tensor with shape [batch_size, z_dim]
       Returns
       -------
       Returns the computed MMD between x and y.
    """
    alphas = [
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
        1e3, 1e4, 1e5, 1e6
    ]
    alphas = torch.autograd.Variable(torch.FloatTensor(alphas)).to(device=source_features.device)

    cost = torch.mean(gaussian_kernel_matrix(source_features, source_features, alphas))
    cost += torch.mean(gaussian_kernel_matrix(target_features, target_features, alphas))
    cost -= 2 * torch.mean(gaussian_kernel_matrix(source_features, target_features, alphas))

    return cost