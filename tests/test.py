import sys

sys.path.append("../")
import cpa
import scanpy as sc
import scvi
from cpa.helper import rank_genes_groups_by_cov


def sim_adata():
    adata = scvi.data.synthetic_iid(run_setup_anndata=False)
    sc.pp.filter_cells(adata, min_counts=0)
    sc.pp.log1p(adata)

    adata.obs["condition"] = "drugA"
    adata.obs["condition"].values[:100] = "control"
    adata.obs["condition"].values[350:400] = "control"
    adata.obs["condition"].values[100:200] = "drugB"
    adata.obs["split"] = "train"

    return adata


if __name__ == "__main__":
    adata = sim_adata()

    cpa_api = cpa.api.API(
        adata,
        pretrained=None,
        perturbation_key="condition",
        dose_key="dose_val",
        covariate_keys=["batch"],
        hparams={},
        device="cuda:0",
        control="control",
    )

    print("\nStart training")
    cpa_api.train(max_epochs=1)
