import pandas as pd
from pathlib import Path
from statistics import mode
from scipy.stats import binom
import re
import numpy as np

# local imports
import sys
sys.path.append(str(Path(__file__).parents[1]))
from utils.plotting import plot_recoveries, plot_descriptive_adequacy
from utils.helper_functions import chance_level, logit, inv_logit


if __name__ == "__main__":
    path = Path(__file__).parent

    fig_path = path / "fig"

    # create the figure path if it does not exist
    if not fig_path.exists():
        fig_path.mkdir()

    # load the simulated data
    n_subs = 2
    sim_path = path / "simulated" / "subj_lvl" / f"ORL_{n_subs}_sub.csv"
    sim_data = pd.read_csv(sim_path)

    # load the recovered data
    rec_path = path / "fit" / "subj_lvl" / "param_rec_subj_lvl.csv"
    rec_data = pd.read_csv(rec_path)

    for col in rec_data.columns[:10]:
        print(col)
        
    for col in sim_data.columns[:10]:
        print(col)


    # get the true and recovered parameters
    param_dict = {
        "a_rew_t": [], 
        "a_pun_t": [],
        "omega_f_t": [],
        "omega_p_t": [],
        "K_t": [],
        "a_rew_r": [], 
        "a_pun_r": [],
        "omega_f_r": [],
        "omega_p_r": [],
        "K_r": [],



    }

    for sub in range(1, n_subs + 1):
        tmp_sim = sim_data[sim_data["sub"] == sub]

        for param in ["a_rew", "a_pun", "omega_p", "omega_f", "K"]:
            param_dict[f"{param}_t"].append(tmp_sim[param].unique()[0])

            recovered_param = rec_data[f"{param}.{sub}"]

            if param in ["omega_p", "omega_f"]:
                param_dict[f"{param}_r"].append(np.mean(recovered_param))
            
            elif param in ["a_rew", "a_pun"]:
                pass
        
            elif param in ["K"]:
                pass




    # plot the recovery of the parameters
    plot_recoveries(
        trues = [param_dict["a_rew_t"],param_dict["a_pun_t"], param_dict["K_t"], param_dict["omega_f_t"], param_dict["omega_p_t"]],
        estimateds = [param_dict["a_rew_r"],param_dict["a_pun_r"], param_dict["K_r"], param_dict["omega_f_r"], param_dict["omega_p_r"]],
        parameter_names = [r"$A_{rew}$", r"$A_{pun}$", r"$K$", r"$\omega_f$", r"$\omega_p$"],
        savepath = fig_path / "subj_lvl_parameter_recovery_ORL.png"
    )









"""

    # load the simulated data
    filename = "ORL_simulated_10_groups_20_sub.csv"
    data = pd.read_csv(path / "simulated" / filename)

    # load the estimated data
    filename_est = f"param_rec_{filename}"
    data_estimated = pd.read_csv(path / "fit" / filename_est)

    # get the true parameters
    
    unique_groups = data["group"].unique()

    mu_a_rew_t = []
    mu_a_pun_t = []
    mu_K_t = []
    mu_omega_f_t = []
    mu_omega_p_t = []

    mu_a_rew_e = []
    mu_a_pun_e = []
    mu_K_e = []
    mu_omega_f_e = []
    mu_omega_p_e = []


    for group in unique_groups:
        data_tmp = data[data["group"] == group]
        mu_a_rew_t.append(data_tmp["mu_a_rew"].unique()[0])
        mu_a_pun_t.append(data_tmp["mu_a_pun"].unique()[0])
        mu_K_t.append(data_tmp["mu_K"].unique()[0])
        mu_omega_f_t.append(data_tmp["mu_omega_f"].unique()[0])
        mu_omega_p_t.append(data_tmp["mu_omega_p"].unique()[0])

        data_tmp = data_estimated[data_estimated["group"] == group]
        mu_a_rew_e.append(data_tmp["mu_a_rew"].mean())
        mu_a_pun_e.append(data_tmp["mu_a_pun"].mean())
        mu_K_e.append(data_tmp["mu_K"].mean())
        mu_omega_f_e.append(data_tmp["mu_omega_f"].mean())
        mu_omega_p_e.append(data_tmp["mu_omega_p"].mean())


    # plot the recovery of the parameters
    plot_recoveries(
        trues = [mu_a_rew_t, mu_a_pun_t, mu_K_t, mu_omega_f_t, mu_omega_p_t],
        estimateds = [mu_a_rew_e, mu_a_pun_e, mu_K_e, mu_omega_f_e, mu_omega_p_e],
        parameter_names=["mu_a_pun", "mu_a_rew", "mu_K", "mu_omega_f", "mu_omega_p"],
        #parameter_names = [r"$\mu A_{rew}$", r"$\mu A_{pun}$", r"$\mu K$", r"$\mu \omega_f$", r"$\mu \omega_p$"],
        savepath = fig_path / "hierachical_parameter_recovery_ORL.png"
    )




    # individual parameters
    # get the true parameters
    a_rew_t = []
    a_pun_t = []
    K_t = []
    omega_f_t = []
    omega_p_t = []
    
    a_rew_e = []
    a_pun_e = []
    K_e = []
    omega_f_e = []
    omega_p_e = []

    choices = []
    pred_choices = []
    groups = []

    for group in unique_groups:
        data_tmp = data[data["group"] == group]
        data_tmp_est = data_estimated[data_estimated["group"] == group]

        # unique subject numbers
        unique_subjects = data_tmp["sub"].unique()

        for sub in unique_subjects:
            groups.append(group)
            data_tmp_sub = data_tmp[data_tmp["sub"] == sub]
            a_rew_t.append(data_tmp_sub["sub_a_rew"].iloc[0])
            a_pun_t.append(data_tmp_sub["sub_a_pun"].iloc[0])
            K_t.append(data_tmp_sub["sub_K"].iloc[0])
            omega_f_t.append(data_tmp_sub["sub_omega_f"].iloc[0])
            omega_p_t.append(data_tmp_sub["sub_omega_p"].iloc[0])
            choices.append(data_tmp_sub["choice"].tolist())
            
            a_rew_e.append(data_tmp_est[f"a_rew.{sub}"].mean())
            a_pun_e.append(data_tmp_est[f"a_pun.{sub}"].mean())
            K_e.append(data_tmp_est[f"K.{sub}"].mean())
            omega_f_e.append(data_tmp_est[f"omega_f.{sub}"].mean())
            omega_p_e.append(data_tmp_est[f"omega_p.{sub}"].mean())
            pred_choices.append([int(mode(data_tmp_est[f"y_pred.{sub}.{trial}"])) for trial in range(1, 100+1)])
    
    # plot the recovery of the parameters
    plot_recoveries(
        trues = [a_rew_t, a_pun_t, K_t, omega_f_t, omega_p_t],
        estimateds = [a_rew_e, a_pun_e, K_e, omega_f_e, omega_p_e],
        parameter_names = [r"$A_{rew}$", r"$A_{pun}$", r"$K$", r"$\omega_f$", r"$\omega_p$"],
        #parameter_names = ["a_pun", "a_rew", "K", "omega_f", "omega_p"],
        savepath = fig_path / "hierachical_parameter_recovery_ORL_individual.png"
    )


    # plot the descriptive adequacy
    plot_descriptive_adequacy(
        choices, 
        pred_choices, 
        groups = groups, 
        chance_level = chance_level(100, p = 0.25)*100,
        savepath = fig_path / "descriptive_adequacy_ORL.png"
        )
    

"""