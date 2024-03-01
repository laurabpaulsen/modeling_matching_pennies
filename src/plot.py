from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_prior_posterior(prior:list, posterior:list, param_names:list, true_values:list = None, savepath = None):
    fig, axes = plt.subplots(1, len(prior), figsize = (10, 5))

    for i, (prior, posterior, param_name) in enumerate(zip(prior, posterior, param_names)):
        axes[i].hist(prior, bins = 30, alpha = 0.5, label = "prior")
        axes[i].hist(posterior, bins = 30, alpha = 0.5, label = "posterior")
        axes[i].set_title(param_name)
        axes[i].legend()

        if true_values:
            axes[i].axvline(true_values[i], color = "black", linestyle = "--")



    if savepath:
        plt.savefig(savepath)



if __name__ in "__main__":
    path = Path(__file__).parents[1]

    path_fits = path / "fits"
    path_simulated = path / "data"
    path_plots = path / "plots"

    if not path_plots.exists():
        path_plots.mkdir()

    # load data
    for tmp_fit_path in path_fits.iterdir():    
        print(tmp_fit_path)
        tmp_fit = pd.read_csv(tmp_fit_path)

        tmp_sim = pd.read_csv(path_simulated / tmp_fit_path.name)
        print(tmp_sim.head())

        priors = [tmp_fit["prior_lr"], tmp_fit["prior_tau"]]
        posteriors = [tmp_fit["posterior_lr"], tmp_fit["posterior_tau"]]
        param_names = ["lr", "tau"]
        true_values = [tmp_sim["learning_rate"].values[0], tmp_sim["inverse_temperature"].values[0]]

        plot_prior_posterior(priors, posteriors, param_names, true_values, savepath = path_plots / f"{tmp_fit_path.stem}.png")

