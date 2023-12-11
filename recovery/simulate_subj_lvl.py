"""
This scipt holds the functions for generating the data for the parameter recovery analysis.

If the script is run directly, it will generate the data and save it to a csv file.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# local imports
import sys
sys.path.append(str(Path(__file__).parents[1]))
from utils.simulate import simulate_ORL
from utils.helper_functions import parse_n_subj

def payoff_structure(path):
    """
    loads the payoff structure from the data folder
    """
    payoff = pd.read_csv(path, usecols=["a_outcome", "b_outcome", "c_outcome", "d_outcome"], sep = ";")
    payoff = payoff.to_numpy()

    # scale by 100
    payoff = payoff/100

    print(payoff.shape)

    return payoff

if __name__ in "__main__":
    path = Path(__file__).parent
        
    n_subjects = parse_n_subj()

    # output path for simulated data
    output_path = path / "simulated" / "subj_lvl" / f"{n_subjects}"
    output_path_no_theta = path / "simulated" / "subj_lvl" / f"{n_subjects}_no_theta"

    # create output path if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    output_path_no_theta.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame()

    payoff = payoff_structure(path = path / "payoff_structure.csv")

    for theta_bool, outpath in zip([True, False], [output_path, output_path_no_theta]):
        for subj in range(n_subjects):
            a_rew = np.random.uniform(0, 1)
            a_pun = np.random.uniform(0, 1)

            K = np.random.uniform(0, 5)
            omega_f = np.random.uniform(-2, 2)
            omega_p = np.random.uniform(-2, 2)

            if theta_bool:
                theta = np.random.uniform(0, 5)
            else:
                theta = 1

            data = simulate_ORL(
                payoff = payoff,
                a_rew = a_rew,
                a_pun = a_pun,
                K = K,
                omega_f = omega_f,
                omega_p = omega_p,
                theta = theta
            )
        
            tmp_df = pd.DataFrame.from_dict(data)
            
            tmp_df["a_rew"] = a_rew
            tmp_df["a_pun"] = a_pun
            tmp_df["K"] = K
            tmp_df["omega_f"] = omega_f
            tmp_df["omega_p"] = omega_p
            tmp_df["theta"] = theta
            tmp_df["sub"] = subj + 1

            df = pd.concat([df, tmp_df])

        df.to_csv(outpath / f"ORL.csv", index=False)


    






