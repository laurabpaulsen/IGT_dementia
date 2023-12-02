"""
This scipt holds the functions for generating the data for the parameter recovery analysis.

If the script is run directly, it will generate the data and save it to a csv file.
"""
import numpy as np
import pandas as pd
from pathlib import Path

# local imports
import sys
sys.path.append(str(Path(__file__).parents[1]))
from utils import simulate_ORL



if __name__ in "__main__":
    path = Path(__file__).parent

    # output path for simulated data
    output_path = path / "simulated" / "subj_lvl"

    # create output path if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    n_subjects = 2
    df = pd.DataFrame()
    for subj in range(n_subjects):
        a_rew = np.random.uniform(0, 1)
        a_pun = np.random.uniform(0, 1)


        K = np.random.uniform(0, 5)
        omega_f = np.random.uniform(0, 5)
        omega_p = np.random.uniform(0, 5)


        data = simulate_ORL(
            a_rew = a_rew,
            a_pun = a_pun,
            K = K,
            omega_f = omega_f,
            omega_p = omega_p
        )
    
        tmp_df = pd.DataFrame.from_dict(data)
        tmp_df["a_rew"] = a_rew
        tmp_df["a_pun"] = a_pun
        tmp_df["K"] = K
        tmp_df["omega_f"] = omega_f
        tmp_df["omega_p"] = omega_p
        tmp_df["subject"] = subj + 1

        df = pd.concat([df, tmp_df])

    df.to_csv(output_path / f"ORL_{n_subjects}_sub.csv", index=False)


    






