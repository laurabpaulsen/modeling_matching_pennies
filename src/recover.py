"""
This python file is used to simulate a rescorla-wagner agent playing the matching pennies game.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import stan


if __name__ in "__main__":
    path = Path(__file__).parent



    model_path = path / "model.stan"

    with open(model_path, "r") as f:
        model_spec = f.read()


    for simulation in (path / "data").iterdir():   
        data = pd.read_csv(simulation)

        data_dict = {
            "N" : len(data),
            "outcomes" : data["reward"].values,
            "choices" : data["seeker_choice"].values
        }

        # fit model
        model = stan.build(model_spec, data = data_dict)
        
        fit = model.sample(
            num_chains = 4, 
            num_samples = 1000,
            num_warmup = 1000)

        # get the estimated parameters
        df = fit.to_frame()

        print(df.head())  


        


