import pandas as pd
from pathlib import Path
from statistics import mode
from scipy.stats import binom
import re
import numpy as np

# local imports
import sys
sys.path.append(str(Path(__file__).parents[1]))
from utils.plotting import plot_recoveries, plot_descriptive_adequacy, plot_posteriors_violin
from utils.helper_functions import chance_level, parse_n_subj


if __name__ == "__main__":
    path = Path(__file__).parent

    fig_path = path / "fig"

    # create the figure path if it does not exist
    if not fig_path.exists():
        fig_path.mkdir()

    # load the simulated data
    n_subs = parse_n_subj()


    for theta_bool in [True, False]:
        if theta_bool:
            parameters = ["a_rew", "a_pun", "omega_p", "omega_f", "K", "theta"]
            sim_path = path / "simulated" / "subj_lvl" / f"{n_subs}" / "ORL.csv"
            rec_path = path / "fit" / "subj_lvl" / f"{n_subs}"
        else:
            parameters = ["a_rew", "a_pun", "omega_p", "omega_f", "K"]
            rec_path = path / "fit" / "subj_lvl" / f"{n_subs}_no_theta"
            sim_path = path / "simulated" / "subj_lvl" / f"{n_subs}_no_theta" / "ORL.csv"
        
        # get the true and recovered parameters
        param_dict = {
            f"{param}_{suffix}": [] for param in parameters for suffix in ["t", "r"]
            }

        true_choices = []
        pred_choices = []

        sim_data = pd.read_csv(sim_path)
        
        for sub in range(1, n_subs + 1):
            tmp_sim = sim_data[sim_data["sub"] == sub]
            rec_data = pd.read_csv(rec_path / f"param_rec_subj_{sub}.csv")


            for param in parameters:
                suffix_t, suffix_r = "t", "r"
                param_dict[f"{param}_{suffix_t}"].append(tmp_sim[param].unique()[0])
                    
                recovered_param = rec_data[f"{param}"]
                    
                param_dict[f"{param}_{suffix_r}"].append(np.mean(recovered_param))

            true_choices.append(tmp_sim["choice"].to_list())

            pred_choices_sub = [mode(rec_data[f"y_pred.{t}"]) for t in range(1, 101)]
            pred_choices.append(pred_choices_sub)

        # list with trues
        trues = [param_dict[f"{param}_t"] for param in parameters]
        estimateds = [param_dict[f"{param}_r"] for param in parameters]

        param_names = ["$A_{rew}$", "$A_{pun}$", "$\omega_p$", "$\omega_f$", "$K$"]

        if theta_bool:
            param_names.append("$\\theta$")

        # plot the recovery of the parameters
        plot_recoveries(
            trues = trues,
            estimateds = estimateds,
            parameter_names = param_names,
            savepath = fig_path / "subj_lvl_parameter_recovery_ORL.png" if theta_bool else fig_path / "subj_lvl_parameter_recovery_ORL_no_theta.png"
        )


        plot_descriptive_adequacy(
            choices = true_choices,
            pred_choices = pred_choices,
            chance_level = chance_level(100, p = 1/4, alpha = 0.05)*100,
            savepath = fig_path / "subj_lvl_descriptive_ORL.png" if theta_bool else fig_path / "subj_lvl_descriptive_ORL_no_theta.png"
        )


"""

            # plot the posteriors
            #plot_posteriors_violin(
            #    posteriors = [rec_data[f"{param}"] for param in parameters],
            #    trues = [tmp_sim[param].unique()[0] for param in parameters],
            #    parameter_names = ["$A_{rew}$", "$A_{pun}$", "$\omega_p$", "$\omega_f$", "$K$", "$\\theta$"],
            #    savepath = fig_path / f"subj_{sub}_posteriors_ORL.png"
            #-)
"""