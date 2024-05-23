import os
import torch
import random
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import multiprocessing
from itertools import product


def plot_result(
    ax,
    path: str,
    plot_kwargs: dict = None,
    median=False,
    std=False,
    cumulative=True,
    regret=True,
    minimize=False,
    xlim=None,
    degree=False,
    exploration=False,
    distance=False,
    sci=False,
):
    # path: main directory of experiments (eg. logs/centrality), label : label experiment to plot
    # exp_dir = os.path.join(path, label)
    d_color = {
        "polynomial": "#1f77b4",
        "diffusion_ard": "#8c564b",
        "polynomial_suminverse": "#e377c2",
        "diffusion": "#7f7f7f",
        "random": "#ff7f0e",
        "local_search": "#2ca02c",
        "klocal_search": "#7f7f7f",
        "random_walk": "#0c0c14",
        "dfs": "#d62728",
        "bfs": "#9467bd",
    }
    d_label = {
        "polynomial": "BO_Poly",
        "polynomial_suminverse": "BO_SumInverse",
        "diffusion_ard": "BO_Diff_ARD",
        "diffusion": "BO_Diff",
        "random": "Random",
        "local_search": "Local Search",
        "klocal_search": "kLocal_Search",
        "random_walk": "k-Random Walk",
        "dfs": "DFS",
        "bfs": "BFS",
    }
    exp_dir = path
    algorithm_name = [
        name
        for name in os.listdir(exp_dir)
        if os.path.isdir(os.path.join(exp_dir, name))
    ]

    if degree or exploration or distance:
        if "random" in algorithm_name:
            algorithm_name.remove("random")
        elif "random_walk" in algorithm_name:
            algorithm_name.remove("random_walk")

    min_max_len = np.inf
    tick_size = 25
    my_font = 28
    for algorithm in algorithm_name:
        alg_dir = os.path.join(exp_dir, algorithm)
        ## Here are in directory with signal png and pt
        plot_kwargs = deepcopy(plot_kwargs) or {}
        try:
            data_path_seeds = [f for f in os.listdir(alg_dir) if ".pt" in f]
        except:
            continue
        data_path_seeds.sort()
        # data_path_seeds = data_path_seeds[:n_diff]
        data_over_seeds = []
        for i, df in enumerate(data_path_seeds):
            data_path = os.path.join(alg_dir, df)
            with open(data_path, "rb") as fp:
                data = torch.load(data_path, map_location="cpu")
                minimize = False
            if "regret" in data.keys() and regret and data["regret"] != None:
                y = -data["regret"].numpy().flatten()  # to maximize negative regret
                minimize = True
            else:
                assert "Y" in data.keys()
                y = data["Y"].numpy().flatten()
            if degree:
                y = np.array(data["degree_list"])
                minimize = False
            elif exploration:
                y = np.array(data["n_explored"])
                minimize = False
            elif distance:
                y = np.array(data["distance"])
                minimize = False
            data_over_seeds.append(y)
        n_data_per_trial = np.array([len(d) for d in data_over_seeds])
        try:
            max_len = max(n_data_per_trial)
        except:
            continue
        if len(np.unique(n_data_per_trial)) > 1:
            # pad as appropriate
            for i, d in enumerate(data_over_seeds):
                data_over_seeds[i] = np.concatenate(
                    (d, d[-1] * np.ones(max_len - d.shape[0]))
                )
        all_data = np.array(data_over_seeds)
        if cumulative:
            y = pd.DataFrame(all_data).cummax(axis=1)
        else:
            y = pd.DataFrame(all_data)
        x = np.arange(all_data.shape[1])
        if median:
            mean = y.median(axis=0)
            lb = y.quantile(q=0.25, axis=0)
            ub = y.quantile(q=0.75, axis=0)
        elif std:
            mean = y.mean(axis=0)
            # standard error
            lb = mean - y.std(axis=0)
            ub = mean + y.std(axis=0)
        else:
            mean = y.mean(axis=0)
            # standard error
            lb = mean - y.std(axis=0) / np.sqrt(all_data.shape[0])
            ub = mean + y.std(axis=0) / np.sqrt(all_data.shape[0])
        if minimize:
            mean = -mean
            lb = -lb
            ub = -ub
        ax.plot(
            x[: xlim + 1],
            mean[: xlim + 1],
            ".-",
            label=d_label[algorithm],
            color=d_color[algorithm],
            **plot_kwargs,
        )
        if "alpha" in plot_kwargs.keys():
            del plot_kwargs["alpha"]
        if "markevery" in plot_kwargs.keys():
            del plot_kwargs["markevery"]
        ax.fill_between(
            x[: xlim + 1],
            lb[: xlim + 1],
            ub[: xlim + 1],
            alpha=0.1,
            color=d_color[algorithm],
            **plot_kwargs,
        )
        ax.plot(
            x[: xlim + 1],
            lb[: xlim + 1],
            "-",
            alpha=0.2,
            color=d_color[algorithm],
            **plot_kwargs,
        )
        ax.plot(
            x[: xlim + 1],
            ub[: xlim + 1],
            "-",
            alpha=0.2,
            color=d_color[algorithm],
            **plot_kwargs,
        )
        min_max_len = min(min_max_len, max_len)
    # ax.legend()
    ax.set_xlabel("# Queries", fontsize=my_font)
    ax.tick_params(axis="both", labelsize=tick_size)
    # ax.set_xlim([0, min_max_len])
    ax.set_xlim([0, xlim])
    if exploration or sci:
        ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 4))  # ax.grid()


def plot(exp_path, xlim):
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    n_rows, n_cols = 1, 3
    title_font = 30
    my_font = 28
    tick_font = 25

    fig, axs = plt.subplots(n_rows, n_cols, layout="tight", figsize=(24, 5))
    plot_result(axs[0], exp_path, median=False, std=False, xlim=xlim)
    plot_result(
        axs[1], exp_path, std=True, cumulative=False, exploration=True, xlim=xlim
    )
    plot_result(axs[2], exp_path, std=True, cumulative=False, distance=True, xlim=xlim)
    handles, labels = axs[0].get_legend_handles_labels()

    axs[0].set_title(f"Search Results ", fontsize=title_font)
    axs[1].set_title("Explored ComboGraph", fontsize=title_font)
    axs[2].set_title("Distance from Start", fontsize=my_font)

    axs[0].set_ylabel("Objective", fontsize=my_font)
    axs[0].set_xlabel("#Queries", fontsize=my_font)
    axs[1].set_xlabel("#Queries", fontsize=my_font)
    axs[2].set_xlabel("#Queries", fontsize=my_font)

    axs[1].set_ylabel("# Combo-nodes", fontsize=my_font)
    axs[2].set_ylabel("# Hops", fontsize=my_font)
    axs[1].set_yscale("log", base=10)
    axs[2].set_yscale("log", base=10)

    fig.legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        frameon=True,
        fancybox=False,
        shadow=False,
        ncol=8,
        fontsize=tick_font - 2,
    )

    plt.tight_layout()
    plt.savefig(f"{exp_path}/Results.pdf", bbox_inches="tight")
    plt.savefig(f"{exp_path}/Results.jpg", bbox_inches="tight")

    # plt.show()
