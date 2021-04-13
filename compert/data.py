# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import warnings
import torch

import numpy as np

warnings.simplefilter(action='ignore', category=FutureWarning)
import scanpy as sc
import pandas as pd

from sklearn.preprocessing import OneHotEncoder

def ranks_to_df(data, key='rank_genes_groups'):
    """Converts an `sc.tl.rank_genes_groups` result into a MultiIndex dataframe.

    You can access various levels of the MultiIndex with `df.loc[[category]]`.

    Params
    ------
    data : `AnnData`
    key : str (default: 'rank_genes_groups')
        Field in `.uns` of data where `sc.tl.rank_genes_groups` result is
        stored.
    """
    d = data.uns[key]
    dfs = []
    for k in d.keys():
        if k == 'params':
            continue
        series = pd.DataFrame.from_records(d[k]).unstack()
        series.name = k
        dfs.append(series)

    return pd.concat(dfs, axis=1)


class Dataset:
    def __init__(self,
                 fname,
                 perturbation_key,
                 dose_key,
                 cell_type_key,
                 split_key='split'):

        data = sc.read(fname)

        self.perturbation_key = perturbation_key
        self.dose_key = dose_key
        self.cell_type_key = cell_type_key
        self.genes = torch.Tensor(data.X.A)

        self.var_names = data.var_names        

        self.pert_categories = np.array(data.obs['cov_drug_dose_name'].values)

        self.de_genes = data.uns['rank_genes_groups_cov']
        self.ctrl = data.obs['control'].values
        self.ctrl_name = list(np.unique(data[data.obs['control'] == 1].obs[self.perturbation_key]))

        self.drugs_names = np.array(data.obs[perturbation_key].values)
        self.dose_names = np.array(data.obs[dose_key].values)

        # get unique drugs
        drugs_names_unique = set()
        for d in self.drugs_names:
            [drugs_names_unique.add(i) for i in d.split("+")]
        self.drugs_names_unique = np.array(list(drugs_names_unique))

        # save encoder for a comparison with Mo's model
        # later we need to remove this part
        encoder_drug = OneHotEncoder(sparse=False)
        encoder_drug.fit(self.drugs_names_unique.reshape(-1, 1))
        
        self.atomic_drugs_dict = dict(zip(self.drugs_names_unique, encoder_drug.transform(
                self.drugs_names_unique.reshape(-1, 1))))

        # get drug combinations
        drugs = []
        for i, comb in enumerate(self.drugs_names):
            drugs_combos = encoder_drug.transform(
                np.array(comb.split("+")).reshape(-1, 1))
            dose_combos = str(data.obs[dose_key].values[i]).split("+")
            for j, d in enumerate(dose_combos):
                if j == 0:
                    drug_ohe = float(d) * drugs_combos[j]
                else:
                    drug_ohe += float(d) * drugs_combos[j]
            drugs.append(drug_ohe)
        self.drugs = torch.Tensor(drugs)

        self.cell_types_names = np.array(data.obs[cell_type_key].values)
        self.cell_types_names_unique = np.unique(self.cell_types_names)

        encoder_ct = OneHotEncoder(sparse=False)
        encoder_ct.fit(self.cell_types_names_unique.reshape(-1, 1))

        self.atomic_сovars_dict = dict(zip(list(self.cell_types_names_unique), encoder_ct.transform(
                self.cell_types_names_unique.reshape(-1, 1))))

        self.cell_types = torch.Tensor(encoder_ct.transform(
            self.cell_types_names.reshape(-1, 1))).float()

        self.num_cell_types = len(self.cell_types_names_unique)
        self.num_genes = self.genes.shape[1]
        self.num_drugs = len(self.drugs_names_unique)

        self.indices = {
            "all": list(range(len(self.genes))),
            "control": np.where(data.obs['control'] == 1)[0].tolist(),
            "treated": np.where(data.obs['control'] != 1)[0].tolist(),
            "train": np.where(data.obs[split_key] == 'train')[0].tolist(),
            "test": np.where(data.obs[split_key] == 'test')[0].tolist(),
            "ood": np.where(data.obs[split_key] == 'ood')[0].tolist()
        }

        atomic_ohe = encoder_drug.transform(
            self.drugs_names_unique.reshape(-1, 1))

        self.drug_dict = {}
        for idrug, drug in enumerate(self.drugs_names_unique):
            i = np.where(atomic_ohe[idrug] == 1)[0][0]
            self.drug_dict[i] = drug

        

    def subset(self, split, condition="all"):
        idx = list(set(self.indices[split]) & set(self.indices[condition]))
        return SubDataset(self, idx)

    def __getitem__(self, i):
        return self.genes[i], self.drugs[i], self.cell_types[i]

    def __len__(self):
        return len(self.genes)


class SubDataset:
    """
    Subsets a `Dataset` by selecting the examples given by `indices`.
    """

    def __init__(self, dataset, indices):
        self.perturbation_key = dataset.perturbation_key
        self.dose_key = dataset.dose_key
        self.covars_key = dataset.cell_type_key

        self.perts_dict = dataset.atomic_drugs_dict
        self.covars_dict = dataset.atomic_сovars_dict

        self.genes = dataset.genes[indices]
        self.drugs = dataset.drugs[indices]
        self.cell_types = dataset.cell_types[indices]

        self.drugs_names = dataset.drugs_names[indices]
        self.pert_categories = dataset.pert_categories[indices]
        self.cell_types_names = dataset.cell_types_names[indices]

        self.var_names = dataset.var_names
        self.de_genes = dataset.de_genes
        self.ctrl_name = dataset.ctrl_name[0]

        self.num_cell_types = dataset.num_cell_types
        self.num_genes = dataset.num_genes
        self.num_drugs = dataset.num_drugs

    def __getitem__(self, i):
        return self.genes[i], self.drugs[i], self.cell_types[i]

    def __len__(self):
        return len(self.genes)


def load_dataset_splits(
        dataset_path,
        perturbation_key,
        dose_key,
        cell_type_key,
        split_key,
        return_dataset=False):

    dataset = Dataset(dataset_path,
                      perturbation_key,
                      dose_key,
                      cell_type_key,
                      split_key)

    splits = {
        "training": dataset.subset("train", "all"),
        "training_control": dataset.subset("train", "control"),
        "training_treated": dataset.subset("train", "treated"),
        "test": dataset.subset("test", "all"),
        "test_control": dataset.subset("test", "control"),
        "test_treated": dataset.subset("test", "treated"),
        "ood": dataset.subset("ood", "all")
    }

    if return_dataset:
        return splits, dataset
    else:
        return splits
