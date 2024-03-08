import pandas as pd
from pathlib import Path
from statistics import mode
import re
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append(str(Path(__file__).parents[2]))
from utils.plotting import plot_recoveries

def plot_recovery_coloured(true_lr, est_lr, true_tau, est_tau, savepath = None):
    """
    This function is used to plot the recovery of the parameters and the color is used to indicate the the true value of the other parameter
    """

    fig, axes = plt.subplots(1, 2, figsize = (10, 5), dpi = 300)

    # plot the learning rate
    axes[0].scatter(true_lr, est_lr, c = true_tau, cmap = "Reds", s = 10)
    axes[0].set_xlabel("True learning rate")
    axes[0].set_ylabel("Estimated learning rate")


    # plot the inverse temperature
    axes[1].scatter(true_tau, est_tau, c = true_lr, cmap = "Reds", s = 10)
    axes[1].set_xlabel("True inverse temperature")
    axes[1].set_ylabel("Estimated inverse temperature")


    # plot colorbar on each axis
    for i, ax in enumerate(axes):
        cbar = plt.colorbar(ax.collections[0], ax = ax)
        label = "True value of inverse temperature" if i == 0 else "True value of learning rate"
        cbar.set_label(label, size = 10)

    # 
    for ax in axes:
        x_lims = ax.get_xlim()
        y_lims = ax.get_ylim()
        ax.plot([y_lims[0], x_lims[1]], [y_lims[0], x_lims[1]], color = "black", linestyle = "dashed")


    if savepath:
        plt.savefig(savepath)



if __name__ == "__main__":
    path = Path(__file__).parents[1]

    fig_path = path / "plots" / "recovery"

    # create the figure path if it does not exist
    if not fig_path.exists():
        fig_path.mkdir(parents = True)

    fit_path = path / "fits" 
    sim_path = path / "data"
        
    true_lr = []
    true_tau = []
    estimated_lr = []
    estimated_tau = []

    for fit in fit_path.iterdir():
        fit_df = pd.read_csv(fit)
        sim_df = pd.read_csv(sim_path / f"{fit.stem}.csv")

        true_lr.append(sim_df["learning_rate"].values[0])
        true_tau.append(sim_df["inverse_temperature"].values[0])

        estimated_lr_tmp = fit_df["posterior_lr"].values
        estimated_tau_tmp = fit_df["posterior_tau"].values

        # get the mode of the posterior
        estimated_lr.append(mode(estimated_lr_tmp))
        estimated_tau.append(mode(estimated_tau_tmp))


    # plot the recovery of the parameters
    plot_recoveries(
        trues = [true_lr, true_tau],
        estimateds = [estimated_lr, estimated_tau],
        parameter_names = ["Learning rate", "Inverse temperature"],
        savepath = fig_path / "parameter_recovery.png",
    )

    # plot the recovery of the parameters with the color indicating the true value of the other parameter
    plot_recovery_coloured(
        true_lr = true_lr,
        est_lr = estimated_lr,
        true_tau = true_tau,
        est_tau = estimated_tau,
        savepath = fig_path / "parameter_recovery_coloured.png",)