import pandas as pd
from pathlib import Path
import re
import numpy as np
from math import erf, sqrt

from scipy.stats import norm

# local imports
import sys
sys.path.append(str(Path(__file__).parents[1]))
from utils.plotting import plot_recoveries, plot_descriptive_adequacy, plot_posteriors_violin
from utils.helper_functions import chance_level, parse_n_subj_groups


def load_simulated(path : Path) -> dict:

    data = {}
    # loop over all csv files in the simulated folder
    for file in path.glob("*.csv"):
        data_tmp = pd.read_csv(file)

        data[int(file.stem.split("_")[-1])] = {
            "data" : data_tmp, 
            "mu_a_rew" : data_tmp["mu_a_rew"].unique()[0],
            "mu_a_pun" : data_tmp["mu_a_pun"].unique()[0],
            "mu_K" : data_tmp["mu_K"].unique()[0],
            "mu_theta" : data_tmp["mu_theta"].unique()[0],
            "mu_omega_f" : data_tmp["mu_omega_f"].unique()[0],
            "mu_omega_p" : data_tmp["mu_omega_p"].unique()[0],
            }

    return data

def load_recovered(path : Path) -> dict:
    data = {}

    # loop over all csv files in the simulated folder
    for f in path.glob("*.csv"):

        try:
            data_tmp = pd.read_csv(
                f,
                usecols = lambda x: x.startswith("y_pred") or x.startswith("mu") 
            )
        except pd.errors.EmptyDataError:
            print(f"File {f} is empty")
            continue

        
        group = int(re.split("_", f.stem)[-1])
        
        # get all columns that start with y_pred
        y_pred_cols = [col for col in data_tmp.columns if col.startswith("y_pred")]

        data[f.stem] = {
            #"data" : data_tmp, 
            "group" : group,
            "mu_a_rew" : data_tmp["mu.1"],
            "mu_a_pun" : data_tmp["mu.2"],
            "mu_K" : data_tmp["mu.3"],
            "mu_omega_f" : data_tmp["mu.4"],
            "mu_omega_p" : data_tmp["mu.5"],
            "y_pred" : data_tmp[y_pred_cols]
            }

    return data

def get_true_recovered(parameters_t : list, parameters_r : list, data_sim : dict, data_rec : dict):

    t = {param: [] for param in parameters_t} # true differences
    r = {param: [] for param in parameters_r} # recovered differences
    
    for key in data_rec.keys():
        group = data_rec[key]["group"]

        # true group differences
        for param in parameters_t:
            t[param].append(data_sim[group][param])

        for param_r in parameters_r:
            tmp_data = data_rec[key][param_r] # getting the parameter samples

            # check if nan, then print
            r[param_r].append(tmp_data.mean())

    return t, r


if __name__ == "__main__":
    path = Path(__file__).parent

    n_subj, n_groups = 40, 100
    

    fig_path = path / "fig"

    # create the figure path if it does not exist
    if not fig_path.exists():
        fig_path.mkdir()

    # load the simulated and recovered data
    data_sim = load_simulated(path / "simulated" / "group_lvl" / f"{n_groups}" / f"{n_subj}")
    data_rec = load_recovered(path / "fit" / "group_lvl_no_compare" / f"{n_groups}" / f"{n_subj}")

    parameters_t = ["mu_a_rew", "mu_a_pun", "mu_K", "mu_omega_f", "mu_omega_p"]
    parameters_r = ["mu_a_rew", "mu_a_pun", "mu_K", "mu_omega_f", "mu_omega_p"]

    # get the true and recovered parameters
    t, r = get_true_recovered(parameters_t, parameters_r, data_sim, data_rec)

    # Extract individual lists for true and recovered parameters
    a_rew_t, a_pun_t, K_t, omega_f_t, omega_p_t = t.values()
    a_rew_r, a_pun_r, K_r, omega_f_r, omega_p_r = r.values()

    # inverse probit transformation
    a_rew_r = norm.cdf(a_rew_r)
    a_pun_r = norm.cdf(a_pun_r)
    K_r = norm.cdf(K_r)*5

    # plot the recovery of the parameters
    plot_recoveries(
        trues = [a_rew_t, a_pun_t, K_t, omega_f_t, omega_p_t,],
        estimateds = [a_rew_r, a_pun_r, K_r, omega_f_r, omega_p_r],
        parameter_names = ["$\mu A_{rew}$", "$\mu A_{pun}$", "$\mu  K$", "$\mu  \omega_F$", "$\mu  \omega_P$"],
        savepath = fig_path / "hierachical_parameter_recovery_ORL_no_compare.png",
    )