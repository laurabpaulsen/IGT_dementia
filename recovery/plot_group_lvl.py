import pandas as pd
from pathlib import Path
import re

# local imports
import sys
sys.path.append(str(Path(__file__).parents[1]))
from utils.plotting import plot_recoveries, plot_descriptive_adequacy
from utils.helper_functions import logit, inv_logit, chance_level

# load probit function
from scipy.stats import norm

def probit(p):
    return norm.ppf(p)

def load_simulated(path : Path) -> dict:

    data = {}
    # loop over all csv files in the simulated folder
    for file in path.glob("*.csv"):
        data_tmp = pd.read_csv(file)

        data[int(file.stem.split("_")[-3])] = {
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
            usecols = ["beta_p.2.1" , "beta_p.2.2" , "beta_p.2.3" ,  "beta_p.2.4"  , "beta_p.2.5", "beta_p.2.6"]
        )
        
        group1 = int(re.split("_", f.stem)[-2])
        group2 = int(re.split("_", f.stem)[-1])
        

        data[f.stem] = {
            #"data" : data_tmp, 
            "group1" : group1,
            "group2" : group2,
            "delta_a_rew" : data_tmp["beta_p.2.1"].mean(),
            "delta_a_pun" : data_tmp["beta_p.2.2" ].mean(),
            "delta_K" : data_tmp["beta_p.2.3"].mean(),
            "delta_theta" : data_tmp[ "beta_p.2.4"].mean(),
            "delta_omega_f" : data_tmp[ "beta_p.2.5"].mean(),
            "delta_omega_p" : data_tmp[ "beta_p.2.6"].mean(),
            }

    return data

if __name__ == "__main__":
    path = Path(__file__).parent

    fig_path = path / "fig"

    # create the figure path if it does not exist
    if not fig_path.exists():
        fig_path.mkdir()

    # load the simulated data
    data_sim = load_simulated(path / "simulated" / "group_lvl")

    # load the recovered data
    data_rec = load_recovered(path / "fit" / "group_lvl")

    # Initialize lists for true and recovered parameters
    parameters_t = ["mu_a_rew", "mu_a_pun", "mu_K", "mu_theta", "mu_omega_f", "mu_omega_p"]
    parameters_r = ["delta_a_rew", "delta_a_pun", "delta_K", "delta_theta", "delta_omega_f", "delta_omega_p"]

    t = {param: [] for param in parameters_t} # true differences
    r = {param: [] for param in parameters_r} # recovered differences
    
    for key in data_rec.keys():
        group_1, group_2 = data_rec[key]["group1"], data_rec[key]["group2"]

        # true group differences
        for param in parameters_t:
            #t[param].append(data_sim[group_1][param] - data_sim[group_2][param]) # figure out which of these is the correct one!
            t[param].append(data_sim[group_2][param] - data_sim[group_1][param]) # figure out which of these is the correct one!

        # recovered group differences
        for param_r in parameters_r:
            tmp_data = data_rec[key][param_r] # getting the parameter samples

            if param_r in ["delta_a_rew", "delta_a_pun", "delta_K", "delta_theta":
                tmp_data = probit(tmp_data)
                if param_r in ["delta_K", "delta_theta"]:
                    tmp_data = tmp_data * 5

            r[param_r].append(tmp_data.mean())
                

    # Extract individual lists for true and recovered parameters
    a_rew_t, a_pun_t, K_t, theta_t, omega_f_t, omega_p_t = t.values()
    a_rew_r, a_pun_r, K_r, theta_r, omega_f_r, omega_p_r = r.values()


    # plot the recovery of the parameters
    plot_recoveries(
        trues = [a_rew_t, a_pun_t, K_t, theta_t, omega_f_t, omega_p_t],
        estimateds = [a_rew_r, a_pun_r, K_r, theta_r, omega_f_r, omega_p_r],
        parameter_names = ["$\delta A_{rew}$", "$\delta A_{pun}$", "$\delta  K$", "$\delta \theta$", "$\delta  \omega_f$", "$\delta  \omega_p$"],
        savepath = fig_path / "hierachical_parameter_recovery_ORL.png"
    )
