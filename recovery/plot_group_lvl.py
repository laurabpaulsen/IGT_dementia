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
        data_tmp = pd.read_csv(
            f,
            usecols = lambda x: x.startswith("y_pred") or x.startswith("delta") 
        )
        
        group1 = int(re.split("_", f.stem)[-2])
        group2 = int(re.split("_", f.stem)[-1])
        
        # get all columns that start with y_pred
        y_pred_cols = [col for col in data_tmp.columns if col.startswith("y_pred")]

        data[f.stem] = {
            #"data" : data_tmp, 
            "group1" : group1,
            "group2" : group2,
            "delta_a_rew" : data_tmp["delta.1"],
            "delta_a_pun" : data_tmp["delta.2"],
            "delta_K" : data_tmp["delta.3"],
            "delta_theta" : data_tmp["delta.4"],
            "delta_omega_f" : data_tmp["delta.5"],
            "delta_omega_p" : data_tmp["delta.6"],
            "y_pred" : data_tmp[y_pred_cols]
            }

    return data

def get_true_recovered(parameters_t : list, parameters_r : list, data_sim : dict, data_rec : dict):

    t = {param: [] for param in parameters_t} # true differences
    r = {param: [] for param in parameters_r} # recovered differences
    
    print(f"keys: {data_rec.keys()}")
    for key in data_rec.keys():
        group_1, group_2 = data_rec[key]["group1"], data_rec[key]["group2"]

        # true group differences
        for param in parameters_t:
            t[param].append(data_sim[group_1][param] - data_sim[group_2][param]) # figure out which of these is the correct one!
            #t[param].append(data_sim[group_2][param] - data_sim[group_1][param]) # figure out which of these is the correct one!
            print((f"param: {param}"))
            print(f"group 1: {data_sim[group_1][param]}")
            print(f"group 2: {data_sim[group_2][param]}")

        print("--------------------------")
        # recovered group differences
        
        for param_r in parameters_r:
            tmp_data = data_rec[key][param_r] # getting the parameter samples

            # check if nan, then print
            r[param_r].append(tmp_data.mean())
            print(f"mean of {param_r}: {tmp_data.mean()}")


    return t, r


if __name__ == "__main__":
    path = Path(__file__).parent

    n_subj, n_groups = parse_n_subj_groups()

    fig_path = path / "fig"

    # create the figure path if it does not exist
    if not fig_path.exists():
        fig_path.mkdir()

    # load the simulated data
    data_sim = load_simulated(path / "simulated" / "group_lvl" / f"{n_groups}" / f"{n_subj}")

    # load the recovered data
    data_rec = load_recovered(path / "fit" / "group_lvl" / f"{n_groups}" / f"{n_subj}")

    # plot posterior predictive checks
    keys = list(data_rec.keys())[0]

    posteriors = [data_rec[keys]["delta_a_rew"], data_rec[keys]["delta_a_pun"], data_rec[keys]["delta_K"], data_rec[keys]["delta_theta"], data_rec[keys]["delta_omega_f"], data_rec[keys]["delta_omega_p"]]


    plot_posteriors_violin(
        posteriors = posteriors,
        parameter_names = ["$\Delta A_{rew}$", "$\Delta A_{pun}$", "$\Delta  K$", "$\Delta \\theta$", "$\Delta  \omega_f$", "$\Delta  \omega_p$"],
        trues = None, 
        savepath = fig_path / "hierachical_posteriors_violin_ORL.png"
    )

    # Initialize lists for true and recovered parameters
    parameters_t = ["mu_a_rew", "mu_a_pun", "mu_K", "mu_theta", "mu_omega_f", "mu_omega_p"]
    parameters_r = ["delta_a_rew", "delta_a_pun", "delta_K", "delta_theta", "delta_omega_f", "delta_omega_p"]

    # get the true and recovered parameters
    t, r = get_true_recovered(parameters_t, parameters_r, data_sim, data_rec)

    # Extract individual lists for true and recovered parameters
    a_rew_t, a_pun_t, K_t, theta_t, omega_f_t, omega_p_t = t.values()
    a_rew_r, a_pun_r, K_r, theta_r, omega_f_r, omega_p_r = r.values()


    # plot the recovery of the parameters
    plot_recoveries(
        trues = [a_rew_t, a_pun_t, K_t, theta_t, omega_f_t, omega_p_t],
        estimateds = [a_rew_r, a_pun_r, K_r, theta_r, omega_f_r, omega_p_r],
        parameter_names = ["$\Delta A_{rew}$", "$\Delta A_{pun}$", "$\Delta  K$", "$\Delta \\theta$", "$\Delta  \omega_f$", "$\Delta  \omega_p$"],
        savepath = fig_path / "hierachical_parameter_recovery_ORL.png"
    )
