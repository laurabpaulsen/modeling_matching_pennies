from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_prior_posterior(prior:list, posterior:list, param_names:list, true_values:list = None, savepath = None):
    fig, axes = plt.subplots(1, len(prior), figsize = (10, 5))

    for i, (prior, posterior, param_name) in enumerate(zip(prior, posterior, param_names)):

        axes[i].hist(prior, bins = 50, alpha = 0.5, label = "Prior")
        axes[i].hist(posterior, bins = 50, alpha = 0.5, label = "Posterior")
        axes[i].set_title(param_name)
        axes[i].legend()

        if true_values:
            axes[i].axvline(true_values[i], color = "black", linestyle = "--")


    if savepath:
        plt.savefig(savepath)

    plt.close()


def plot_values(hiders_choices, values, title = None, savepath = None):
    fig, ax = plt.subplots(1, 1, figsize = (10, 5), dpi = 300)


    n_trials = len(hiders_choices)

    ax.plot(range(n_trials), hiders_choices, linewidth = 1, label = "Hider choice", color = "lightblue")
    ax.plot(range(n_trials), values.mean(axis = 0)[:n_trials], linewidth = 1, label = "Value associated with right choice", color = "darkblue")

    ax.legend()

    if title:
        ax.set_title(title)

    if savepath:
        plt.savefig(savepath)
    plt.close()


if __name__ in "__main__":
    path = Path(__file__).parents[1]

    path_fits = path / "fits"
    path_simulated = path / "data"
    path_plots = path / "plots" / "checks"

    if not path_plots.exists():
        path_plots.mkdir(parents = True)

    # load data
    for tmp_fit_path in path_fits.iterdir():    
        tmp_fit = pd.read_csv(tmp_fit_path)
        tmp_sim = pd.read_csv(path_simulated / tmp_fit_path.name)

        priors = [tmp_fit["prior_lr"], tmp_fit["prior_tau"]]
        posteriors = [tmp_fit["posterior_lr"], tmp_fit["posterior_tau"]]
        param_names = ["lr", "tau"]
        true_values = [tmp_sim["learning_rate"].values[0], tmp_sim["inverse_temperature"].values[0]]

        plot_prior_posterior(priors, posteriors, param_names, true_values, savepath = path_plots / f"{tmp_fit_path.stem}.png")


        hider_choices = tmp_sim["hider_choice"].values
        # get all columns that contain the word "value"
        value_columns = [col for col in tmp_fit.columns if "value" in col]

        plot_values(
            hider_choices, 
            tmp_fit[value_columns].values, 
            title = f"Learning rate: {tmp_sim['learning_rate'].values[0].round(3)}, Inverse temperature: {tmp_sim['inverse_temperature'].values[0].round(3)}",
            savepath = path_plots / f"{tmp_fit_path.stem}_values.png")

        
    
