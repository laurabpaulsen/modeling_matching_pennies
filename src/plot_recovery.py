import pandas as pd
from pathlib import Path
from statistics import mode
import re
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append(str(Path(__file__).parents[2]))
from utils.plotting import plot_recoveries


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