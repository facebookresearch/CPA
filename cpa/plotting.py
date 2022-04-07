# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from collections import defaultdict

import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from sklearn.decomposition import KernelPCA
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import cosine_similarity

FONT_SIZE = 10
font = {"size": FONT_SIZE}

matplotlib.rc("font", **font)
matplotlib.rc("ytick", labelsize=FONT_SIZE)
matplotlib.rc("xtick", labelsize=FONT_SIZE)


class CPAVisuals:
    """
    A wrapper for automatic plotting CompPert latent embeddings and dose-response
    curve. Sets up prefix for all files and default dictionaries for atomic
    perturbations and cell types.
    """

    def __init__(
        self,
        cpa,
        fileprefix=None,
        perts_palette=None,
        covars_palette=None,
        plot_params={"fontsize": None},
    ):
        """
        Parameters
        ----------
        cpa : CompPertAPI
            Variable from API class.
        fileprefix : str, optional (default: None)
            Prefix (with path) to the filename to save all embeddings in a
            standartized manner. If None, embeddings are not saved to file.
        perts_palette : dict (default: None)
            Dictionary of colors for the embeddings of perturbations. Keys
            correspond to perturbations and values to their colors. If None,
            default dicitonary will be set up.
        covars_palette : dict (default: None)
            Dictionary of colors for the embeddings of covariates. Keys
            correspond to covariates and values to their colors. If None,
            default dicitonary will be set up.
        """

        self.fileprefix = fileprefix

        self.perturbation_key = cpa.perturbation_key
        self.dose_key = cpa.dose_key
        self.covariate_keys = cpa.covariate_keys
        self.measured_points = cpa.measured_points

        self.unique_perts = cpa.unique_perts
        self.unique_covars = cpa.unique_covars

        if perts_palette is None:
            self.perts_palette = dict(
                zip(self.unique_perts, get_palette(len(self.unique_perts)))
            )
        else:
            self.perts_palette = perts_palette

        if covars_palette is None:
            self.covars_palette = {}
            for cov in self.unique_covars:
                self.covars_palette[cov] = dict(
                    zip(
                        self.unique_covars[cov],
                        get_palette(len(self.unique_covars[cov]), palette_name="tab10"),
                    )
                )
        else:
            self.covars_palette = covars_palette

        if plot_params["fontsize"] is None:
            self.fontsize = FONT_SIZE
        else:
            self.fontsize = plot_params["fontsize"]

    def plot_latent_embeddings(
        self,
        emb,
        titlename="Example",
        kind="perturbations",
        palette=None,
        labels=None,
        dimred="KernelPCA",
        filename=None,
        show_text=True,
    ):
        """
        Parameters
        ----------
        emb : np.array
            Multi-dimensional embedding of perturbations or covariates.
        titlename : str, optional (default: 'Example')
            Title.
        kind : int, optional, optional (default: 'perturbations')
            Specify if this is embedding of perturbations, covariates or some
            other. If it is perturbations or covariates, it will use default
            saved dictionaries for colors.
        palette : dict, optional (default: None)
            If embedding of kind not perturbations or covariates, the user can
            specify color dictionary for the embedding.
        labels : list, optional (default: None)
            Labels for the embeddings.
        dimred : str, optional (default: 'KernelPCA')
            Dimensionality reduction method for plotting low dimensional
            representations. Options: 'KernelPCA', 'UMAPpre', 'UMAPcos', None.
            If None, uses first 2 dimensions of the embedding.
        filename : str (default: None)
            Name of the file to save the plot. If None, will automatically
            generate name from prefix file.
        """
        if filename is None:
            if self.fileprefix is None:
                filename = None
                file_name_similarity = None
            else:
                filename = f"{self.fileprefix}_emebdding.png"
                file_name_similarity = f"{self.fileprefix}_emebdding_similarity.png"
        else:
            file_name_similarity = filename.split(".")[0] + "_similarity.png"

        if labels is None:
            if kind == "perturbations":
                palette = self.perts_palette
                labels = self.unique_perts
            elif kind in self.unique_covars:
                palette = self.covars_palette[kind]
                labels = self.unique_covars[kind]

        if len(emb) < 2:
            print(f"Embedding contains only {len(emb)} vectors. Not enough to plot.")
        else:
            plot_embedding(
                fast_dimred(emb, method=dimred),
                labels,
                show_lines=True,
                show_text=show_text,
                col_dict=palette,
                title=titlename,
                file_name=filename,
                fontsize=self.fontsize,
            )

            plot_similarity(
                emb,
                labels,
                col_dict=palette,
                fontsize=self.fontsize,
                file_name=file_name_similarity,
            )

    def plot_contvar_response2D(
        self,
        df_response2D,
        df_ref=None,
        levels=15,
        figsize=(4, 4),
        xlims=(0, 1.03),
        ylims=(0, 1.03),
        palette="coolwarm",
        response_name="response",
        title_name=None,
        fontsize=None,
        postfix="",
        filename=None,
        alpha=0.4,
        sizes=(40, 160),
        logdose=False,
    ):

        """
        Parameters
        ----------
        df_response2D : pd.DataFrame
            Data frame with responses of combinations with columns=(dose1, dose2,
            response).
        levels: int, optional (default: 15)
            Number of levels for contour plot.
        response_name : str (default: 'response')
            Name of column in df_response to plot as response.
        alpha: float (default: 0.4)
            Transparency of the background contour.
        figsize: tuple (default: (4,4))
            Size of the figure in inches.
        palette : dict, optional (default: None)
            Colors dictionary for perturbations to plot.
        title_name : str, optional (default: None)
            Title for the plot.
        postfix : str, optional (defualt: '')
            Postfix to add to the output file name to save the model.
        filename : str, optional (defualt: None)
            Name of the file to save the plot.  If None, will automatically
            generate name from prefix file.
        logdose: bool (default: False)
            If True, dose values will be log10. 0 values will be mapped to
            minumum value -1,e.g.
            if smallest non-zero dose was 0.001, 0 will be mapped to -4.
        """
        sns.set_style("white")

        if (filename is None) and not (self.fileprefix is None):
            filename = f"{self.fileprefix}_{postfix}response2D.png"
        if fontsize is None:
            fontsize = self.fontsize

        x_name, y_name = df_response2D.columns[:2]

        x = df_response2D[x_name].values
        y = df_response2D[y_name].values

        if logdose:
            x = log10_with0(x)
            y = log10_with0(y)

        z = df_response2D[response_name].values

        n = int(np.sqrt(len(x)))

        X = x.reshape(n, n)
        Y = y.reshape(n, n)
        Z = z.reshape(n, n)

        fig, ax = plt.subplots(figsize=figsize)

        CS = ax.contourf(X, Y, Z, cmap=palette, levels=levels, alpha=alpha)
        CS = ax.contour(X, Y, Z, levels=15, cmap=palette)
        ax.clabel(CS, inline=1, fontsize=fontsize)
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("equal")
        ax.axis("square")
        ax.yaxis.set_tick_params(labelsize=fontsize)
        ax.xaxis.set_tick_params(labelsize=fontsize)
        ax.set_xlabel(x_name, fontsize=fontsize, fontweight="bold")
        ax.set_ylabel(y_name, fontsize=fontsize, fontweight="bold")
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)

        # sns.despine(left=False, bottom=False, right=True)
        sns.despine()

        if not (df_ref is None):
            sns.scatterplot(
                x=x_name,
                y=y_name,
                hue="split",
                size="num_cells",
                sizes=sizes,
                alpha=1.0,
                palette={"train": "#000000", "training": "#000000", "ood": "#e41a1c"},
                data=df_ref,
                ax=ax,
            )
            ax.legend_.remove()

        ax.set_title(title_name, fontweight="bold", fontsize=fontsize)
        plt.tight_layout()

        if filename:
            save_to_file(fig, filename)

    def plot_contvar_response(
        self,
        df_response,
        response_name="response",
        var_name=None,
        df_ref=None,
        palette=None,
        title_name=None,
        postfix="",
        xlabelname=None,
        filename=None,
        logdose=False,
        fontsize=None,
        measured_points=None,
        bbox=(1.35, 1.0),
        figsize=(7.0, 4.0),
    ):
        """
        Parameters
        ----------
        df_response : pd.DataFrame
            Data frame of responses.
        response_name : str (default: 'response')
            Name of column in df_response to plot as response.
        var_name : str, optional  (default: None)
            Name of conditioning variable, e.g. could correspond to covariates.
        df_ref : pd.DataFrame, optional  (default: None)
            Reference values. Fields for plotting should correspond to
            df_response.
        palette : dict, optional (default: None)
            Colors dictionary for perturbations to plot.
        title_name : str, optional (default: None)
            Title for the plot.
        postfix : str, optional (defualt: '')
            Postfix to add to the output file name to save the model.
        filename : str, optional (defualt: None)
            Name of the file to save the plot.  If None, will automatically
            generate name from prefix file.
        logdose: bool (default: False)
            If True, dose values will be log10. 0 values will be mapped to
            minumum value -1,e.g.
            if smallest non-zero dose was 0.001, 0 will be mapped to -4.
        figsize: tuple (default: (7., 4.))
            Size of output figure
        """
        if (filename is None) and not (self.fileprefix is None):
            filename = f"{self.fileprefix}_{postfix}response.png"

        if fontsize is None:
            fontsize = self.fontsize

        if logdose:
            dose_name = f"log10-{self.dose_key}"
            df_response[dose_name] = log10_with0(df_response[self.dose_key].values)
            if not (df_ref is None):
                df_ref[dose_name] = log10_with0(df_ref[self.dose_key].values)
        else:
            dose_name = self.dose_key

        if var_name is None:
            if len(self.unique_covars) > 1:
                var_name = self.covars_key
            else:
                var_name = self.perturbation_key

        if palette is None:
            if var_name == self.perturbation_key:
                palette = self.perts_palette
            elif var_name in self.covariate_keys:
                palette = self.covars_palette[var_name]

        plot_dose_response(
            df_response,
            dose_name,
            var_name,
            xlabelname=xlabelname,
            df_ref=df_ref,
            response_name=response_name,
            title_name=title_name,
            use_ref_response=(not (df_ref is None)),
            col_dict=palette,
            plot_vertical=False,
            f1=figsize[0],
            f2=figsize[1],
            fname=filename,
            logscale=measured_points,
            measured_points=measured_points,
            bbox=bbox,
            fontsize=fontsize,
            figformat="png",
        )

    def plot_scatter(
        self,
        df,
        x_axis,
        y_axis,
        hue=None,
        size=None,
        style=None,
        figsize=(4.5, 4.5),
        title=None,
        palette=None,
        filename=None,
        alpha=0.75,
        sizes=(30, 90),
        text_dict=None,
        postfix="",
        fontsize=14,
    ):

        sns.set_style("white")

        if (filename is None) and not (self.fileprefix is None):
            filename = f"{self.fileprefix}_scatter{postfix}.png"

        if fontsize is None:
            fontsize = self.fontsize

        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
        sns.scatterplot(
            x=x_axis,
            y=y_axis,
            hue=hue,
            style=style,
            size=size,
            sizes=sizes,
            alpha=alpha,
            palette=palette,
            data=df,
        )

        ax.legend_.remove()
        ax.set_xlabel(x_axis, fontsize=fontsize)
        ax.set_ylabel(y_axis, fontsize=fontsize)
        ax.xaxis.set_tick_params(labelsize=fontsize)
        ax.yaxis.set_tick_params(labelsize=fontsize)
        ax.set_title(title)
        if not (text_dict is None):
            texts = []
            for label in text_dict.keys():
                texts.append(
                    ax.text(
                        text_dict[label][0],
                        text_dict[label][1],
                        label,
                        fontsize=fontsize,
                    )
                )

            adjust_text(
                texts, arrowprops=dict(arrowstyle="-", color="black", lw=0.1), ax=ax
            )

        plt.tight_layout()

        if filename:
            save_to_file(fig, filename)


def log10_with0(x):
    mx = np.min(x[x > 0])
    x[x == 0] = mx / 10
    return np.log10(x)


def get_palette(n_colors, palette_name="Set1"):

    try:
        palette = sns.color_palette(palette_name)
    except:
        print("Palette not found. Using default palette tab10")
        palette = sns.color_palette()
    while len(palette) < n_colors:
        palette += palette

    return palette


def fast_dimred(emb, method="KernelPCA"):
    """
    Takes high dimensional embeddings and produces a 2-dimensional representation
    for plotting.
    emb: np.array
        Embeddings matrix.
    method: str (default: 'KernelPCA')
        Method for dimensionality reduction: KernelPCA, UMAPpre, UMAPcos, tSNE.
        If None return first 2 dimensions of the embedding vector.
    """
    if method is None:
        return emb[:, :2]
    elif method == "KernelPCA":
        similarity_matrix = cosine_similarity(emb)
        np.fill_diagonal(similarity_matrix, 1.0)
        X = KernelPCA(n_components=2, kernel="precomputed").fit_transform(
            similarity_matrix
        )
    else:
        raise NotImplementedError

    return X


def plot_dose_response(
    df,
    contvar_key,
    perturbation_key,
    df_ref=None,
    response_name="response",
    use_ref_response=False,
    palette=None,
    col_dict=None,
    fontsize=8,
    measured_points=None,
    interpolate=True,
    f1=7,
    f2=3.0,
    bbox=(1.35, 1.0),
    ref_name="origin",
    title_name="None",
    plot_vertical=True,
    fname=None,
    logscale=None,
    xlabelname=None,
    figformat="png",
):

    """Plotting decoding of the response with respect to dose.

    Params
    ------
    df : `DataFrame`
        Table with columns=[perturbation_key, contvar_key, response_name].
        The last column is always "response".
    contvar_key : str
        Name of the column in df for values to use for x axis.
    perturbation_key : str
        Name of the column in df for the perturbation or covariate to plot.
    response_name: str (default: response)
        Name of the column in df for values to use for y axis.
    df_ref : `DataFrame` (default: None)
        Table with the same columns as in df to plot ground_truth or another
        condition for comparison. Could
        also be used to just extract reference values for x-axis.
    use_ref_response : bool (default: False)
        A flag indicating if to use values for y axis from df_ref (True) or j
        ust to extract reference values for x-axis.
    col_dict : dictionary (default: None)
        Dictionary with colors for each value in perturbation_key.
    bbox : tuple (default: (1.35, 1.))
        Coordinates to adjust the legend.
    plot_vertical : boolean (default: False)
        Flag if to plot reference values for x axis from df_ref dataframe.
    f1 : float (default: 7.0))
        Width in inches for the plot.
    f2 : float (default: 3.0))
        Hight in inches for the plot.
    fname : str (default: None)
        Name of the file to export the plot. The name comes without format
        extension.
    format : str (default: png)
        Format for the file to export the plot.
    """
    sns.set_style("white")
    if use_ref_response and not (df_ref is None):
        df[ref_name] = "predictions"
        df_ref[ref_name] = "observations"
        if interpolate:
            df_plt = pd.concat([df, df_ref])
        else:
            df_plt = df
    else:
        df_plt = df
    df_plt = df_plt.reset_index()

    atomic_drugs = np.unique(df[perturbation_key].values)
    if palette is None:
        current_palette = get_palette(len(list(atomic_drugs)))

    if col_dict is None:
        col_dict = dict(zip(list(atomic_drugs), current_palette))

    fig = plt.figure(figsize=(f1, f2))
    ax = plt.gca()
    if use_ref_response:
        sns.lineplot(
            x=contvar_key,
            y=response_name,
            palette=col_dict,
            hue=perturbation_key,
            style=ref_name,
            dashes=[(1, 0), (2, 1)],
            legend="full",
            style_order=["predictions", "observations"],
            data=df_plt,
            ax=ax,
        )

        df_ref = df_ref.replace("training_treated", "train")
        sns.scatterplot(
            x=contvar_key,
            y=response_name,
            hue="split",
            size="num_cells",
            sizes=(10, 100),
            alpha=1.0,
            palette={"train": "#000000", "training": "#000000", "ood": "#e41a1c"},
            data=df_ref,
            ax=ax,
        )
        sns.despine()
        ax.legend_.remove()
    else:
        sns.lineplot(
            x=contvar_key,
            y=response_name,
            palette=col_dict,
            hue=perturbation_key,
            data=df_plt,
            ax=ax,
        )
        ax.legend(loc="upper right", bbox_to_anchor=bbox, fontsize=fontsize)
        sns.despine()
    if not (title_name is None):
        ax.set_title(title_name, fontsize=fontsize, fontweight="bold")
    ax.grid("off")

    if xlabelname is None:
        ax.set_xlabel(contvar_key, fontsize=fontsize)
    else:
        ax.set_xlabel(xlabelname, fontsize=fontsize)

    ax.set_ylabel(f"{response_name}", fontsize=fontsize)

    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)

    if not (logscale is None):
        ax.set_xticks(np.log10(logscale))
        ax.set_xticklabels(logscale, rotation=90)

    if not (df_ref is None):
        atomic_drugs = np.unique(df_ref[perturbation_key].values)
        for drug in atomic_drugs:
            x = df_ref[df_ref[perturbation_key] == drug][contvar_key].values
            m1 = np.min(df[df[perturbation_key] == drug][response_name].values)
            m2 = np.max(df[df[perturbation_key] == drug][response_name].values)

            if plot_vertical:
                for x_dot in x:
                    ax.plot(
                        [x_dot, x_dot],
                        [m1, m2],
                        ":",
                        color="black",
                        linewidth=0.5,
                        alpha=0.5,
                    )
    fig.tight_layout()
    if fname:
        plt.savefig(f"{fname}.{figformat}", format=figformat, dpi=600)

    return fig


def plot_uncertainty_comb_dose(
    cpa_api,
    cov,
    pert,
    N=11,
    metric="cosine",
    measured_points=None,
    cond_key="condition",
    figsize=(4, 4),
    vmin=None,
    vmax=None,
    sizes=(40, 160),
    df_ref=None,
    xlims=(0, 1.03),
    ylims=(0, 1.03),
    fixed_drugs="",
    fixed_doses="",
    title="",
    filename=None,
):
    """Plotting uncertainty for a single perturbation at a dose range for a
    particular covariate.

    Params
    ------
    cpa_api
        Api object for the model class.
    cov : dict
        Name of covariate.
    pert : str
        Name of the perturbation.
    N : int
        Number of dose values.
    metric: str (default: 'cosine')
        Metric to evaluate uncertainty.
    measured_points : dict (default: None)
        A dicitionary of dictionaries. Per each covariate a dictionary with
        observed doses per perturbation, e.g. {'covar1': {'pert1':
        [0.1, 0.5, 1.0], 'pert2': [0.3]}
    cond_key : str (default: 'condition')
        Name of the variable to use for plotting.
    filename : str (default: None)
        Full path to the file to export the plot. File extension should be
        included.

    Returns
        -------
        pd.DataFrame of uncertainty estimations.
    """

    cov_name = "_".join([cov[cov_key] for cov_key in cpa_api.covariate_keys])
    df_list = []
    for i in np.round(np.linspace(0, 1, N), decimals=2):
        for j in np.round(np.linspace(0, 1, N), decimals=2):
            df_list.append(
                {
                    "covariates": cov_name,
                    "condition": pert + fixed_drugs,
                    "dose_val": str(i) + "+" + str(j) + fixed_doses,
                }
            )
    df_pred = pd.DataFrame(df_list)
    uncert_cos = []
    uncert_eucl = []
    closest_cond_cos = []
    closest_cond_eucl = []
    for i in range(df_pred.shape[0]):
        (
            uncert_cos_,
            uncert_eucl_,
            closest_cond_cos_,
            closest_cond_eucl_,
        ) = cpa_api.compute_uncertainty(
            cov=cov, pert=df_pred.iloc[i]["condition"], dose=df_pred.iloc[i]["dose_val"]
        )
        uncert_cos.append(uncert_cos_)
        uncert_eucl.append(uncert_eucl_)
        closest_cond_cos.append(closest_cond_cos_)
        closest_cond_eucl.append(closest_cond_eucl_)

    df_pred["uncertainty_cosine"] = uncert_cos
    df_pred["uncertainty_eucl"] = uncert_eucl
    df_pred["closest_cond_cos"] = closest_cond_cos
    df_pred["closest_cond_eucl"] = closest_cond_eucl
    doses = df_pred.dose_val.apply(lambda x: x.split("+"))
    X = np.array(doses.apply(lambda x: x[0]).astype(float)).reshape(N, N)
    Y = np.array(doses.apply(lambda x: x[1]).astype(float)).reshape(N, N)
    Z = np.array(df_pred[f"uncertainty_{metric}"].values.astype(float)).reshape(N, N)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    CS = ax.contourf(X, Y, Z, cmap="coolwarm", levels=20, alpha=1, vmin=vmin, vmax=vmax)

    ax.set_xlabel(pert.split("+")[0], fontweight="bold")
    ax.set_ylabel(pert.split("+")[1], fontweight="bold")

    if not (df_ref is None):
        sns.scatterplot(
            x=pert.split("+")[0],
            y=pert.split("+")[1],
            hue="split",
            size="num_cells",
            sizes=sizes,
            alpha=1.0,
            palette={"train": "#000000", "training": "#000000", "ood": "#e41a1c"},
            data=df_ref,
            ax=ax,
        )
        ax.legend_.remove()

    if measured_points:
        ticks = measured_points[cov_name][pert]
        xticks = [float(x.split("+")[0]) for x in ticks]
        yticks = [float(x.split("+")[1]) for x in ticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks, rotation=90)
        ax.set_yticks(yticks)
    fig.colorbar(CS)
    sns.despine()
    ax.axis("equal")
    ax.axis("square")
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_title(title, fontsize=10, fontweight='bold')
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=600)

    return df_pred


def plot_uncertainty_dose(
    cpa_api,
    cov,
    pert,
    N=11,
    metric="cosine",
    measured_points=None,
    cond_key="condition",
    log=False,
    min_dose=None,
    filename=None,
):
    """Plotting uncertainty for a single perturbation at a dose range for a
    particular covariate.

    Params
    ------
    cpa_api
        Api object for the model class.
    cov : str
        Name of covariate.
    pert : str
        Name of the perturbation.
    N : int
        Number of dose values.
    metric: str (default: 'cosine')
        Metric to evaluate uncertainty.
    measured_points : dict (default: None)
        A dicitionary of dictionaries. Per each covariate a dictionary with
        observed doses per perturbation, e.g. {'covar1': {'pert1':
        [0.1, 0.5, 1.0], 'pert2': [0.3]}
    cond_key : str (default: 'condition')
        Name of the variable to use for plotting.
    log : boolean (default: False)
        A flag if to plot on a log scale.
    min_dose : float (default: None)
        Minimum dose for the uncertainty estimate.
    filename : str (default: None)
        Full path to the file to export the plot. File extension should be included.

    Returns
        -------
        pd.DataFrame of uncertainty estimations.
    """

    df_list = []
    if log:
        if min_dose is None:
            min_dose = 1e-3
        N_val = np.round(np.logspace(np.log10(min_dose), np.log10(1), N), decimals=10)
    else:
        if min_dose is None:
            min_dose = 0
        N_val = np.round(np.linspace(min_dose, 1.0, N), decimals=3)

    cov_name = "_".join([cov[cov_key] for cov_key in cpa_api.covariate_keys])

    for i in N_val:
        df_list.append({"covariates": cov_name, "condition": pert, "dose_val": repr(i)})

    df_pred = pd.DataFrame(df_list)
    uncert_cos = []
    uncert_eucl = []
    closest_cond_cos = []
    closest_cond_eucl = []

    for i in range(df_pred.shape[0]):
        (
            uncert_cos_,
            uncert_eucl_,
            closest_cond_cos_,
            closest_cond_eucl_,
        ) = cpa_api.compute_uncertainty(
            cov=cov, pert=df_pred.iloc[i]["condition"], dose=df_pred.iloc[i]["dose_val"]
        )
        uncert_cos.append(uncert_cos_)
        uncert_eucl.append(uncert_eucl_)
        closest_cond_cos.append(closest_cond_cos_)
        closest_cond_eucl.append(closest_cond_eucl_)

    df_pred["uncertainty_cosine"] = uncert_cos
    df_pred["uncertainty_eucl"] = uncert_eucl
    df_pred["closest_cond_cos"] = closest_cond_cos
    df_pred["closest_cond_eucl"] = closest_cond_eucl

    x = df_pred.dose_val.values.astype(float)
    y = df_pred[f"uncertainty_{metric}"].values.astype(float)
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, y)
    ax.set_xlabel(pert)
    ax.set_ylabel("Uncertainty")
    ax.set_title(cov_name)
    if log:
        ax.set_xscale("log")
    if measured_points:
        ticks = measured_points[cov_name][pert]
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks, rotation=90)
    else:
        plt.draw()
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    sns.despine()
    plt.tight_layout()

    if filename:
        plt.savefig(filename)

    return df_pred


def save_to_file(fig, file_name, file_format=None):
    if file_format is None:
        if file_name.split(".")[-1] in ["png", "pdf"]:
            file_format = file_name.split(".")[-1]
            savename = file_name
        else:
            file_format = "pdf"
            savename = f"{file_name}.{file_format}"
    else:
        savename = file_name

    fig.savefig(savename, format=file_format, dpi=600)
    print(f"Saved file to: {savename}")


def plot_embedding(
    emb,
    labels=None,
    col_dict=None,
    title=None,
    show_lines=False,
    show_text=False,
    show_legend=True,
    axis_equal=True,
    circle_size=40,
    circe_transparency=1.0,
    line_transparency=0.8,
    line_width=1.0,
    fontsize=9,
    fig_width=4,
    fig_height=4,
    file_name=None,
    file_format=None,
    labels_name=None,
    width_ratios=[7, 1],
    bbox=(1.3, 0.7),
):
    sns.set_style("white")

    # create data structure suitable for embedding
    df = pd.DataFrame(emb, columns=["dim1", "dim2"])
    if not (labels is None):
        if labels_name is None:
            labels_name = "labels"
        df[labels_name] = labels

    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = plt.gca()

    sns.despine(left=False, bottom=False, right=True)

    if (col_dict is None) and not (labels is None):
        col_dict = get_colors(labels)

    sns.scatterplot(
        x="dim1",
        y="dim2",
        hue=labels_name,
        palette=col_dict,
        alpha=circe_transparency,
        edgecolor="none",
        s=circle_size,
        data=df,
        ax=ax,
    )

    try:
        ax.legend_.remove()
    except:
        pass

    if show_lines:
        for i in range(len(emb)):
            if col_dict is None:
                ax.plot(
                    [0, emb[i, 0]],
                    [0, emb[i, 1]],
                    alpha=line_transparency,
                    linewidth=line_width,
                    c=None,
                )
            else:
                ax.plot(
                    [0, emb[i, 0]],
                    [0, emb[i, 1]],
                    alpha=line_transparency,
                    linewidth=line_width,
                    c=col_dict[labels[i]],
                )

    if show_text and not (labels is None):
        texts = []
        labels = np.array(labels)
        unique_labels = np.unique(labels)
        for label in unique_labels:
            idx_label = np.where(labels == label)[0]
            texts.append(
                ax.text(
                    np.mean(emb[idx_label, 0]),
                    np.mean(emb[idx_label, 1]),
                    label,
                    #fontsize=fontsize,
                )
            )

        adjust_text(
            texts, arrowprops=dict(arrowstyle="-", color="black", lw=0.1), ax=ax
        )

    if axis_equal:
        ax.axis("equal")
        ax.axis("square")

    if title:
        ax.set_title(title, fontweight="bold")

    ax.set_xlabel("dim1"),# fontsize=fontsize)
    ax.set_ylabel("dim2"),# fontsize=fontsize)
    #ax.xaxis.set_tick_params(labelsize=fontsize)
    #ax.yaxis.set_tick_params(labelsize=fontsize)

    plt.tight_layout()

    if file_name:
        save_to_file(fig, file_name, file_format)

    return plt


def get_colors(labels, palette=None, palette_name=None):
    n_colors = len(labels)
    if palette is None:
        palette = get_palette(n_colors, palette_name)
    col_dict = dict(zip(labels, palette[:n_colors]))
    return col_dict


def plot_similarity(
    emb,
    labels=None,
    col_dict=None,
    fig_width=4,
    fig_height=4,
    cmap="coolwarm",
    fmt="png",
    fontsize=7,
    file_format=None,
    file_name=None,
):

    # first we take construct similarity matrix
    # add another similarity
    similarity_matrix = cosine_similarity(emb)

    df = pd.DataFrame(
        similarity_matrix,
        columns=labels,
        index=labels,
    )

    if col_dict is None:
        col_dict = get_colors(labels)

    network_colors = pd.Series(df.columns, index=df.columns).map(col_dict)

    sns_plot = sns.clustermap(
        df,
        cmap=cmap,
        center=0,
        row_colors=network_colors,
        col_colors=network_colors,
        mask=False,
        metric="euclidean",
        figsize=(fig_height, fig_width),
        vmin=-1,
        vmax=1,
        fmt=file_format,
    )

    sns_plot.ax_heatmap.xaxis.set_tick_params(labelsize=fontsize)
    sns_plot.ax_heatmap.yaxis.set_tick_params(labelsize=fontsize)
    sns_plot.ax_heatmap.axis("equal")
    sns_plot.cax.yaxis.set_tick_params(labelsize=fontsize)

    if file_name:
        save_to_file(sns_plot, file_name, file_format)


from scipy import sparse, stats
from sklearn.metrics import r2_score


def mean_plot(
    adata,
    pred,
    condition_key,
    exp_key,
    path_to_save="./reg_mean.pdf",
    gene_list=None,
    deg_list=None,
    show=False,
    title=None,
    verbose=False,
    x_coeff=0.30,
    y_coeff=0.8,
    fontsize=11,
    R2_type="R2",
    figsize=(3.5, 3.5),
    **kwargs,
):
    """
    Plots mean matching.

    # Parameters
    adata: `~anndata.AnnData`
        Contains real v
    pred: `~anndata.AnnData`
        Contains predicted values.
    condition_key: Str
        adata.obs key to look for x-axis and y-axis condition
    exp_key: Str
        Condition in adata.obs[condition_key] to be ploted
    path_to_save: basestring
        Path to save the plot.
    gene_list: list
        List of gene names to be plotted.
    deg_list: list
        List of DEGs to compute R2
    show: boolean
        if True plots the figure
    Verbose: boolean
        If true prints the value
    title: Str
        Title of the plot
    x_coeff: float
        Shifts R2 text horizontally by x_coeff
    y_coeff: float
        Shifts R2 text vertically by y_coeff
    show: bool
        if `True`: will show to the plot after saving it.
    fontsize: int
        Font size for R2 texts
    R2_type: Str
        How to compute R2 value, should be either Pearson R2 or R2 (sklearn)

    Returns:
    Calluated R2 values
    """

    r2_types = ["R2", "Pearson R2"]
    if R2_type not in r2_types:
        raise ValueError("R2 caclulation should be one of" + str(r2_types))
    if sparse.issparse(adata.X):
        adata.X = adata.X.A
    if sparse.issparse(pred.X):
        pred.X = pred.X.A
    diff_genes = deg_list
    real = adata[adata.obs[condition_key] == exp_key]
    pred = pred[pred.obs[condition_key] == exp_key]
    if diff_genes is not None:
        if hasattr(diff_genes, "tolist"):
            diff_genes = diff_genes.tolist()
        real_diff = adata[:, diff_genes][adata.obs[condition_key] == exp_key]
        pred_diff = pred[:, diff_genes][pred.obs[condition_key] == exp_key]
        x_diff = np.average(pred_diff.X, axis=0)
        y_diff = np.average(real_diff.X, axis=0)
        if R2_type == "R2":
            r2_diff = r2_score(y_diff, x_diff)
        if R2_type == "Pearson R2":
            m, b, pearson_r_diff, p_value_diff, std_err_diff = stats.linregress(
                y_diff, x_diff
            )
            r2_diff = pearson_r_diff ** 2
        if verbose:
            print(f"Top {len(diff_genes)} DEGs var: ", r2_diff)
    x = np.average(pred.X, axis=0)
    y = np.average(real.X, axis=0)
    if R2_type == "R2":
        r2 = r2_score(y, x)
    if R2_type == "Pearson R2":
        m, b, pearson_r, p_value, std_err = stats.linregress(y, x)
        r2 = pearson_r ** 2
    if verbose:
        print("All genes var: ", r2)
    df = pd.DataFrame({f"{exp_key}_true": x, f"{exp_key}_pred": y})

    plt.figure(figsize=figsize)
    ax = sns.regplot(x=f"{exp_key}_true", y=f"{exp_key}_pred", data=df)
    ax.tick_params(labelsize=fontsize)
    if "range" in kwargs:
        start, stop, step = kwargs.get("range")
        ax.set_xticks(np.arange(start, stop, step))
        ax.set_yticks(np.arange(start, stop, step))
    ax.set_xlabel("true", fontsize=fontsize)
    ax.set_ylabel("pred", fontsize=fontsize)
    if gene_list is not None:
        for i in gene_list:
            j = adata.var_names.tolist().index(i)
            x_bar = x[j]
            y_bar = y[j]
            plt.text(x_bar, y_bar, i, fontsize=fontsize, color="black")
            plt.plot(x_bar, y_bar, "o", color="red", markersize=5)
    if title is None:
        plt.title(f"", fontsize=fontsize, fontweight="bold")
    else:
        plt.title(title, fontsize=fontsize, fontweight="bold")
    ax.text(
        max(x) - max(x) * x_coeff,
        max(y) - y_coeff * max(y),
        r"$\mathrm{R^2_{\mathrm{\mathsf{all\ genes}}}}$= " + f"{r2:.2f}",
        fontsize=fontsize,
    )
    if diff_genes is not None:
        ax.text(
            max(x) - max(x) * x_coeff,
            max(y) - (y_coeff + 0.15) * max(y),
            r"$\mathrm{R^2_{\mathrm{\mathsf{DEGs}}}}$= " + f"{r2_diff:.2f}",
            fontsize=fontsize,
        )
    plt.savefig(f"{path_to_save}", bbox_inches="tight", dpi=100)
    if show:
        plt.show()
    plt.close()
    if diff_genes is not None:
        return r2, r2_diff
    else:
        return r2


def plot_r2_matrix(pred, adata, de_genes=None, **kwds):
    """Plots a pairwise R2 heatmap between predicted and control conditions.

    Params
    ------
    pred : `AnnData`
        Must have the field `cov_drug_dose_name`
    adata : `AnnData`
        Original gene expression data, with the field `cov_drug_dose_name`.
    de_genes : `dict`
        Dictionary of de_genes, where the keys
        match the categories in `cov_drug_dose_name`
    """
    r2s_mean = defaultdict(list)
    r2s_var = defaultdict(list)
    conditions = pred.obs["cov_drug_dose_name"].cat.categories
    for cond in conditions:
        if de_genes:
            degs = de_genes[cond]
            y_pred = pred[:, degs][pred.obs["cov_drug_dose_name"] == cond].X
            y_true_adata = adata[:, degs]
        else:
            y_pred = pred[pred.obs["cov_drug_dose_name"] == cond].X
            y_true_adata = adata

        # calculate r2 between pairwise
        for cond_real in conditions:
            y_true = y_true_adata[
                y_true_adata.obs["cov_drug_dose_name"] == cond_real
            ].X.toarray()
            r2s_mean[cond_real].append(
                r2_score(y_true.mean(axis=0), y_pred.mean(axis=0))
            )
            r2s_var[cond_real].append(r2_score(y_true.var(axis=0), y_pred.var(axis=0)))

    for r2_dict in [r2s_mean, r2s_var]:
        r2_df = pd.DataFrame.from_dict(r2_dict, orient="index")
        r2_df.columns = conditions

        plt.figure(figsize=(5, 5))
        p = sns.heatmap(
            data=r2_df,
            vmin=max(r2_df.min(0).min(), 0),
            cmap="Blues",
            cbar=False,
            annot=True,
            fmt=".2f",
            annot_kws={"fontsize": 5},
            **kwds,
        )
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
        plt.xlabel("y_true")
        plt.ylabel("y_pred")
        plt.show()


def arrange_history(history):

    print(history.keys())


class CPAHistory:
    """
    A wrapper for automatic plotting history of CPA model..
    """

    def __init__(self, cpa_api, fileprefix=None):
        """
        Parameters
        ----------
        cpa_api : dict
            cpa api instance
        fileprefix : str, optional (default: None)
            Prefix (with path) to the filename to save all embeddings in a
            standartized manner. If None, embeddings are not saved to file.
        """
        self.history = cpa_api.history
        self.time = self.history["elapsed_time_min"]
        self.losses_list = [
            "loss_reconstruction",
            "loss_adv_drugs",
            "loss_adv_covariates",
        ]
        self.penalties_list = ["penalty_adv_drugs", "penalty_adv_covariates"]

        subset_keys = ["epoch"] + self.losses_list + self.penalties_list

        self.losses = pd.DataFrame(
            dict((k, self.history[k]) for k in subset_keys if k in self.history)
        )

        self.header = ["mean", "mean_DE", "var", "var_DE"]
        self.eval_metrics = False
        if 'perturbation disentanglement' in list (self.history):               #check that metrics were evaluated
            self.eval_metrics = True
            self.metrics = pd.DataFrame(columns=["epoch", "split"] + self.header)
            for split in ["training", "test", "ood"]:
                df_split = pd.DataFrame(np.array(self.history[split]), columns=self.header)
                df_split["split"] = split
                df_split["epoch"] = self.history["stats_epoch"]
                self.metrics = pd.concat([self.metrics, df_split])
            self.covariate_names = list(cpa_api.datasets["training"].covariate_names)
            self.disent = pd.DataFrame(
                dict(
                    (k, self.history[k])
                    for k in 
                    ['perturbation disentanglement'] 
                    + [f'{cov} disentanglement' for cov in self.covariate_names]
                    if k in self.history
                )
            )
            self.disent["epoch"] = self.history["stats_epoch"]
        self.fileprefix = fileprefix

    def print_time(self):
        print(f"Computation time: {self.time:.0f} min")

    def plot_losses(self, filename=None):
        """
        Parameters
        ----------
        filename : str (default: None)
            Name of the file to save the plot. If None, will automatically
            generate name from prefix file.
        """
        if filename is None:
            if self.fileprefix is None:
                filename = None
            else:
                filename = f"{self.fileprefix}_history_losses.png"

        fig, ax = plt.subplots(1, 4, sharex=True, sharey=False, figsize=(12, 3))

        i = 0
        for i in range(4):
            if i < 3:
                ax[i].plot(
                    self.losses["epoch"].values, self.losses[self.losses_list[i]].values
                )
                ax[i].set_title(self.losses_list[i], fontweight="bold")
            else:
                ax[i].plot(
                    self.losses["epoch"].values, self.losses[self.penalties_list].values
                )
                ax[i].set_title("Penalties", fontweight="bold")
        sns.despine()
        plt.tight_layout()

        if filename:
            save_to_file(fig, filename)

    def plot_r2_metrics(self, epoch_min=0, filename=None):
        """
        Parameters
        ----------
        epoch_min : int (default: 0)
            Epoch from which to show metrics history plot. Done for readability.

        filename : str (default: None)
            Name of the file to save the plot. If None, will automatically
            generate name from prefix file.
        """

        assert self.eval_metrics == True, 'The evaluation metrics were not computed'

        if filename is None:
            if self.fileprefix is None:
                filename = None
            else:
                filename = f"{self.fileprefix}_history_metrics.png"

        df = self.metrics.melt(id_vars=["epoch", "split"])
        col_dict = dict(
            zip(["training", "test", "ood"], ["#377eb8", "#4daf4a", "#e41a1c"])
        )
        fig, axs = plt.subplots(2, 2, sharex=True, sharey=False, figsize=(7, 5))
        ax = plt.gca()
        i = 0
        for i1 in range(2):
            for i2 in range(2):
                sns.lineplot(
                    data=df[
                        (df["variable"] == self.header[i]) & (df["epoch"] > epoch_min)
                    ],
                    x="epoch",
                    y="value",
                    palette=col_dict,
                    hue="split",
                    ax=axs[i1, i2],
                )
                axs[i1, i2].set_title(self.header[i], fontweight="bold")
                i += 1
        sns.despine()
        plt.tight_layout()
        if filename:
            save_to_file(fig, filename)

    def plot_disentanglement_metrics(self, epoch_min=0, filename=None):
        """
        Parameters
        ----------
        epoch_min : int (default: 0)
            Epoch from which to show metrics history plot. Done for readability.

        filename : str (default: None)
            Name of the file to save the plot. If None, will automatically
            generate name from prefix file.
        """
        assert self.eval_metrics == True, 'The evaluation metrics were not computed'
        
        if filename is None:
            if self.fileprefix is None:
                filename = None
            else:
                filename = f"{self.fileprefix}_history_metrics.png"

        fig, axs = plt.subplots(
            1, 
            1+len(self.covariate_names), 
            sharex=True, 
            sharey=False, 
            figsize=(2 + 5*(len(self.covariate_names)), 3)
        )

        ax = plt.gca()
        sns.lineplot(
            data=self.disent[self.disent["epoch"] > epoch_min],
            x="epoch",
            y="perturbation disentanglement",
            legend=False,
            ax=axs[0],
        )
        axs[0].set_title("perturbation disent", fontweight="bold")

        for i, cov in enumerate(self.covariate_names):
            sns.lineplot(
                data=self.disent[self.disent['epoch'] > epoch_min],
                x="epoch",
                y=f"{cov} disentanglement",
                legend=False,
                ax=axs[1+i]
            )
            axs[1+i].set_title(f"{cov} disent", fontweight="bold")
        fig.tight_layout()
        sns.despine()