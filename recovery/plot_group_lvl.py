import pandas as pd
from pathlib import Path
from statistics import mode
from scipy.stats import binom
import re

# local imports
import sys
sys.path.append(str(Path(__file__).parents[1]))
from plot_fns import plot_recoveries, plot_descriptive_adequacy
from utils import logit, inv_logit


def chance_level(n, alpha = 0.001, p = 0.5):
    """
    Calculates the chance level for a given number of trials and alpha level

    Parameters
    ----------
    n : int
        The number of trials.
    alpha : float
        The alpha level.
    p : float
        The probability of a correct response.

    Returns
    -------
    chance_level : float
        The chance level.
    """
    k = binom.ppf(1-alpha, n, p)
    chance_level = k/n
    
    return chance_level


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
            usecols=["a_pun_betas.1", "a_rew_betas.1", "K_betas.1", "omega_f_betas.1", "omega_p_betas.1",  # intercepts
                     "a_pun_betas.2", "a_rew_betas.2", "K_betas.2", "omega_f_betas.2", "omega_p_betas.2"]) # slopes
        
        group1 = int(re.split("_", file.stem)[-2])
        group2 = int(re.split("_", file.stem)[-1])
        

        data[file.stem] = {
            #"data" : data_tmp, 
            "group1" : group1,
            "group2" : group2,
            "intercept_a_pun" : data_tmp["a_pun_betas.1"].mean(),
            "intercept_a_rew" : data_tmp["a_rew_betas.1"].mean(),
            "intercept_K" : data_tmp["K_betas.1"].mean(),
            "intercept_omega_f" : data_tmp["omega_f_betas.1"].mean(),
            "intercept_omega_p" : data_tmp["omega_p_betas.1"].mean(),
            "slope_a_pun" : data_tmp["a_pun_betas.2"].mean(),
            "slope_a_rew" : data_tmp["a_rew_betas.2"].mean(),
            "slope_K" : data_tmp["K_betas.2"].mean(),
            "slope_omega_f" : data_tmp["omega_f_betas.2"].mean(),
            "slope_omega_p" : data_tmp["omega_p_betas.2"].mean(),
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
        group_1 = data_rec[key]["group1"] # check if this is correct or if it should be the other way around
        group_2 = data_rec[key]["group2"] # check if this is correct or if it should be the other way around

        # true group differences
        mu_a_rew_t.append(data_sim[group_1]["mu_a_rew"] - data_sim[group_2]["mu_a_rew"])
        mu_a_pun_t.append(data_sim[group_1]["mu_a_pun"] - data_sim[group_2]["mu_a_pun"])
        mu_K_t.append(data_sim[group_1]["mu_K"] - data_sim[group_2]["mu_K"])
        mu_omega_f_t.append(data_sim[group_1]["mu_omega_f"] - data_sim[group_2]["mu_omega_f"])
        mu_omega_p_t.append(data_sim[group_1]["mu_omega_p"] - data_sim[group_2]["mu_omega_p"])

        # recovered group differences
        # ADD THE CALCULATION OF THE GROUP DIFFERENCES HERE!!! 
        # NEED TO TAKE INTO ACCOUNT THE INTERCEPTS AND THE SLOPES and logit transformation of the 

        """
        K = inv_logit(X * K_betas +  K_subj[subject]); // restriciting K to be between 0 and 5
        omega_f = X * omega_f_betas + omega_f_subj[subject];
        omega_p = X * omega_p_betas + omega_p_subj[subject];
        a_rew = inv_logit(X * a_rew_betas + a_rew_subj[subject]);
        a_pun = inv_logit(X * a_pun_betas + a_pun_subj[subject]);
        """
        recovered_mu_K = logit(data_rec[key]["slope_K"])
        recovered_mu_omega_f =  data_rec[key]["slope_omega_f"]
        recovered_mu_omega_p = data_rec[key]["slope_omega_p"]
        recovered_mu_a_rew = logit(data_rec[key]["slope_a_rew"])
        recovered_mu_a_pun = logit(data_rec[key]["slope_a_pun"])

        print(recovered_mu_K, recovered_mu_omega_f, recovered_mu_omega_p, recovered_mu_a_rew, recovered_mu_a_pun)

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