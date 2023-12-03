import pandas as pd
from pathlib import Path
import re

# local imports
import sys
sys.path.append(str(Path(__file__).parents[1]))
from utils.plotting import plot_recoveries, plot_descriptive_adequacy
from utils.helper_functions import logit, inv_logit, chance_level




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
            "mu_omega_f" : data_tmp["mu_omega_f"].unique()[0],
            "mu_omega_p" : data_tmp["mu_omega_p"].unique()[0],

            }

    return data

def load_recovered(path : Path) -> dict:
    data = {}

    # loop over all csv files in the simulated folder
    for file in path.glob("*.csv"):
        data_tmp = pd.read_csv(
            file, 
            usecols=[
                "delta_a_pun", "delta_a_rew", "delta_K", "delta_omega_f", "delta_omega_p", # slopes
                # implement choices here!
            ] 
            ) 
        
        group1 = int(re.split("_", file.stem)[-2])
        group2 = int(re.split("_", file.stem)[-1])
        

        data[file.stem] = {
            #"data" : data_tmp, 
            "group1" : group1,
            "group2" : group2,
            "delta_a_rew" : data_tmp["delta_a_rew"].mean(),
            "delta_a_pun" : data_tmp["delta_a_pun"].mean(),
            "delta_K" : data_tmp["delta_K"].mean(),
            "delta_omega_f" : data_tmp["delta_omega_f"].mean(),
            "delta_omega_p" : data_tmp["delta_omega_p"].mean(),
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

    # get the true and recovered parameters
    mu_a_rew_t = []
    mu_a_pun_t = []
    mu_K_t = []
    mu_omega_f_t = []
    mu_omega_p_t = []

    mu_a_rew_r = []
    mu_a_pun_r = []
    mu_K_r = []
    mu_omega_f_r = []
    mu_omega_p_r = []


    for key in data_rec.keys():
        
        # group 1 is modelled as -0.5 and group 2 as 0.5
        group_1 = data_rec[key]["group2"] # check if this is correct or if it should be the other way around
        group_2 = data_rec[key]["group1"] # check if this is correct or if it should be the other way around

        # true group differences
        mu_a_rew_t.append(data_sim[group_1]["mu_a_rew"] - data_sim[group_2]["mu_a_rew"])
        mu_a_pun_t.append(data_sim[group_1]["mu_a_pun"] - data_sim[group_2]["mu_a_pun"])
        mu_K_t.append(data_sim[group_1]["mu_K"] - data_sim[group_2]["mu_K"])
        mu_omega_f_t.append(data_sim[group_1]["mu_omega_f"] - data_sim[group_2]["mu_omega_f"])
        mu_omega_p_t.append(data_sim[group_1]["mu_omega_p"] - data_sim[group_2]["mu_omega_p"])

        # recovered group differences
        # ADD THE CALCULATION OF THE GROUP DIFFERENCES HERE!!! 
        # NEED TO TAKE INTO ACCOUNT THE INTERCEPTS AND THE SLOPES and logit transformation of the 
        recovered_mu_K = data_rec[key]["delta_K"]
        recovered_mu_omega_f =  data_rec[key]["delta_omega_f"]
        recovered_mu_omega_p = data_rec[key]["delta_omega_p"]
        recovered_mu_a_rew = data_rec[key]["delta_a_rew"]
        recovered_mu_a_pun = data_rec[key]["delta_a_pun"]


        mu_a_rew_r.append(recovered_mu_a_rew)
        mu_a_pun_r.append(recovered_mu_a_pun)
        mu_K_r.append(recovered_mu_K)
        mu_omega_f_r.append(recovered_mu_omega_f)
        mu_omega_p_r.append(recovered_mu_omega_p)


    # plot the recovery of the parameters
    plot_recoveries(
        trues = [mu_a_rew_t, mu_a_pun_t, mu_K_t, mu_omega_f_t, mu_omega_p_t],
        estimateds = [mu_a_rew_r, mu_a_pun_r, mu_K_r, mu_omega_f_r, mu_omega_p_r],
        parameter_names=["mu_a_pun", "mu_a_rew", "mu_K", "mu_omega_f", "mu_omega_p"],
        #parameter_names = [r"$\mu A_{rew}$", r"$\mu A_{pun}$", r"$\mu K$", r"$\mu \omega_f$", r"$\mu \omega_p$"],
        savepath = fig_path / "hierachical_parameter_recovery_ORL.png"
    )
