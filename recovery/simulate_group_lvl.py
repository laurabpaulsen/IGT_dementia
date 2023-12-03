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

if __name__ in "__main__":
    path = Path(__file__).parent

    # output path for simulated data
    output_path = path / "simulated" / "group_lvl"

    # create output path if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    n_groups = 20
    n_subjects = 20

    for group in range(n_groups):
        mu_a_rew = np.random.uniform(0, 1)
        mu_a_pun = np.random.uniform(0, 1)


        mu_K = np.random.uniform(0, 5)
        mu_omega_f = np.random.uniform(0, 5)
        mu_omega_p = np.random.uniform(0, 5)


        data = simulate_ORL_group(
            n_subjects = n_subjects,
            mu_a_rew = mu_a_rew,
            mu_a_pun = mu_a_pun,
            mu_K = mu_K,
            mu_omega_f = mu_omega_f,
            mu_omega_p = mu_omega_p,
            sigma_a_rew = 0.05,
            sigma_a_pun = 0.05,
            sigma_K = 0.05,
            sigma_omega_f = 0.05,
            sigma_omega_p = 0.05
            )
    
        df = pd.DataFrame.from_dict(data)
        df["mu_a_rew"] = mu_a_rew
        df["mu_a_pun"] = mu_a_pun
        df["mu_K"] = mu_K
        df["mu_omega_f"] = mu_omega_f
        df["mu_omega_p"] = mu_omega_p


        df.to_csv(output_path / f"ORL_simulated_group_{group+1}_{n_subjects}_sub.csv", index=False)


    






