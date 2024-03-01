"""
This python file is used to simulate a rescorla-wagner agent playing the matching pennies game.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import stan


if __name__ in "__main__":
    path = Path(__file__).parents[1]
    model_path = path / "model.stan"
    outpath = path / "fits"
    
    if not outpath.exists():
        outpath.mkdir()

    with open(model_path, "r") as f:
        model_spec = f.read()

    for simulation in (path / "data").iterdir():   
        data = pd.read_csv(simulation)

        data_dict = {
            "T" : int(len(data)),
            "outcomes" : data["reward"].astype("int").values,
            "choices" : data["seeker_choice"].astype("int").values,
            "prior_sd_lr": 1,
            "prior_sd_tau": 0.5,
        }

        # fit model
        model = stan.build(model_spec, data = data_dict)
        
        fit = model.sample(
            num_chains = 4, 
            num_samples = 1000,
            num_warmup = 1000)

        # get the estimated parameters
        df = fit.to_frame()

        df.to_csv(outpath / f"{simulation.stem}.csv", index = False)


        


