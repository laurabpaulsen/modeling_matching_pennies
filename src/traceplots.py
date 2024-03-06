import pandas as pd
from pathlib import Path
from statistics import mode
import re
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append(str(Path(__file__).parents[2]))
from utils.plotting import plot_traceplots, plot_trankplot


if __name__ in "__main__":
    path = Path(__file__).parents[1]
    fig_path = path / "plots" / "traceplots"

    if not fig_path.exists():
        fig_path.mkdir(parents = True)

    # get the data
    est_data = pd.read_csv(path / "fits" / "simulation_1.csv")

    parameters = ["lr", "tau"]
    # convert the data to the correct format
    traceplot_data = est_data[parameters]
    traceplot_data = traceplot_data.to_dict(orient = "list")
    
    # split each key into 4 list (one for each chain)
    traceplot_data = {key: [traceplot_data[key][i::4] for i in range(4)] for key in traceplot_data.keys()}

    plot_traceplots(traceplot_data, parameters, savepath = fig_path / f"traceplot{1}.png")

    plot_trankplot(traceplot_data, parameters, savepath = fig_path / f"rankplot{1}.png")
