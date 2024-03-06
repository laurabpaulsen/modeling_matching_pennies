"""
This python file is used to simulate a rescorla-wagner agent playing the matching pennies game.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def softmax_binary(x, theta):
    """
    Softmax function.

    Parameters
    ----------
    x : numpy.ndarray
        Value
    theta : float
        Inverse temperature parameter.
    
    Returns
    -------
    softmax : numpy.ndarray
        Softmaxed values.
    """

    return 1 / (1 + np.exp(-theta*x))



class RescorlaWagnerAgent:
    def __init__(self, learning_rate, inverse_temperature, initial_value = 0.5):
        self.alpha = learning_rate
        self.inv_temp = inverse_temperature
        self.value = initial_value

    def update_value(self, reward, choice):
        if choice == 1:
            self.value = self.value + self.alpha * (reward - self.value)
        else:
            self.value = self.value - self.alpha * (reward - (1-self.value))

    def choose_action(self):
        p_right = softmax_binary(self.value - 0.5, self.inv_temp)

        if np.random.uniform() < p_right:
            return 1
        else:
            return 0

    
class RandomAgent:
    def __init__(self, bias):
        self.bias = bias

    def choose_action(self):
        if np.random.uniform() < self.bias:
            return 1
        else:
            return 0


if __name__ in "__main__":

    path = Path(__file__).parents[1]

    data_path = path / "data"
    data_path.mkdir(exist_ok=True)

    for sim in range(50):
        # generate random values
        seeker_inv_temp = np.random.uniform(0.4, 3)
        seeker_learning_rate = np.random.uniform(0, 1)
        n_trials = 120

        seeker = RescorlaWagnerAgent(seeker_learning_rate, seeker_inv_temp)
        hider = RandomAgent(0.8)

        df = pd.DataFrame(columns=["seeker_choice", "hider_choice", "reward", "seeker_value"])
        
        for t in range(n_trials):
            seeker_action = seeker.choose_action()
            hider_action = hider.choose_action()
            if seeker_action == hider_action:
                reward = 1
            else:
                reward = 0
            
            seeker.update_value(reward, seeker_action)

            df.loc[t] = [seeker_action, hider_action, reward, seeker.value]

        df["learning_rate"] = seeker.alpha
        df["inverse_temperature"] = seeker.inv_temp
        df["trial"] = np.arange(n_trials)
        df["n_trials"] = n_trials


        filename = f"simulation_{sim+1}.csv"
        df.to_csv(data_path / filename, index=False)



