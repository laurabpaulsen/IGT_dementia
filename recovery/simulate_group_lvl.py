"""
This scipt holds the functions for generating the data for the parameter recovery analysis.

If the script is run directly, it will generate the data and save it to a csv file.
"""
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parents[1]))
from utils.simulate import simulate_ORL_group
from utils.helper_functions import parse_n_subj_groups


def payoff_structure(path):
    """
    loads the payoff structure from the data folder
    """
    payoff = pd.read_csv(path, usecols=["a_outcome", "b_outcome", "c_outcome", "d_outcome"], sep = ";")
    payoff = payoff.to_numpy()

    # scale by 100
    payoff = payoff/100

    return payoff

if __name__ in "__main__":
    path = Path(__file__).parent

    n_subjects, n_groups = parse_n_subj_groups()

    # output path for simulated data
    output_path = path / "simulated" / "group_lvl" / f"{n_groups}" / f"{n_subjects}"

    # create output path if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # load the payoff structure
    payoff = payoff_structure(path / "payoff_structure.csv")

    for group in range(n_groups):
        mu_a_rew = np.random.uniform(0, 1)
        mu_a_pun = np.random.uniform(0, 1)

        mu_K = np.random.uniform(0, 5)
        mu_omega_f = np.random.uniform(-2, 2)
        mu_omega_p = np.random.uniform(-2, 2)
        mu_theta = 1

        data = simulate_ORL_group(
            payoff = payoff,
            n_subjects = n_subjects,
            mu_a_rew = mu_a_rew,
            mu_a_pun = mu_a_pun,
            mu_K = mu_K,
            mu_omega_f = mu_omega_f,
            mu_omega_p = mu_omega_p,
            mu_theta = mu_theta,
            sigma_a_rew = 0.05,
            sigma_a_pun = 0.05,
            sigma_K = 0.05,
            sigma_omega_f = 0.05,
            sigma_omega_p = 0.05,
            sigma_theta = 0.05
            )
    
        df = pd.DataFrame.from_dict(data)
        df["mu_a_rew"] = mu_a_rew
        df["mu_a_pun"] = mu_a_pun
        df["mu_K"] = mu_K
        df["mu_omega_f"] = mu_omega_f
        df["mu_omega_p"] = mu_omega_p
        df["mu_theta"] = mu_theta


        df.to_csv(output_path / f"ORL_simulated_group_{group+1}.csv", index=False)


    






