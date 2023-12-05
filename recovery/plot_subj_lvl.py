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
from utils.helper_functions import parse_n_subj


if __name__ == "__main__":
    path = Path(__file__).parent

    fig_path = path / "fig"

    # create the figure path if it does not exist
    if not fig_path.exists():
        fig_path.mkdir()

    # load the simulated data
    n_subs = parse_n_subj()
    sim_path = path / "simulated" / "subj_lvl" / f"ORL_{n_subs}_sub.csv"
    sim_data = pd.read_csv(sim_path)

    # load the recovered data
    rec_path = path / "fit" / "subj_lvl" 
    
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

    true_choices = []
    pred_choices = []
    
    for sub in range(1, n_subs + 1):
        tmp_sim = sim_data[sim_data["sub"] == sub]
        rec_data = pd.read_csv(rec_path / f"param_rec_subj_{sub}.csv")

        for param in ["a_rew", "a_pun", "omega_p", "omega_f", "K"]:
            suffix_t, suffix_r = "t", "r"
            param_dict[f"{param}_{suffix_t}"].append(tmp_sim[param].unique()[0])
            
            recovered_param = rec_data[f"{param}"]
            if param in ["a_rew", "a_pun", "k"]:
                recovered_param = inv_logit(recovered_param)
                if param == "k":
                    recovered_param = recovered_param * 5
            param_dict[f"{param}_{suffix_r}"].append(np.mean(recovered_param))

        true_choices.append(tmp_sim["choice"].to_list())

        pred_choices_sub = [mode(rec_data[f"y_pred.{t}"]) for t in range(1, 101)]
        pred_choices.append(pred_choices_sub)

    # plot the recovery of the parameters
    plot_recoveries(
        trues = [param_dict["a_rew_t"],param_dict["a_pun_t"], param_dict["K_t"], param_dict["omega_f_t"], param_dict["omega_p_t"]],
        estimateds = [param_dict["a_rew_r"],param_dict["a_pun_r"], param_dict["K_r"], param_dict["omega_f_r"], param_dict["omega_p_r"]],
        parameter_names = [r"$A_{rew}$", r"$A_{pun}$", r"$K$", r"$\omega_f$", r"$\omega_p$"],
        savepath = fig_path / "subj_lvl_parameter_recovery_ORL.png"
    )


    plot_descriptive_adequacy(
        choices = true_choices,
        pred_choices = pred_choices,
        chance_level = chance_level(100, p = 0.25)*100,
        savepath =fig_path / "subj_lvl_descriptive_ORL.png"
    )
