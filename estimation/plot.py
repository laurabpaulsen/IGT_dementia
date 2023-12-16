import pandas as pd
from pathlib import Path
from statistics import mode
import numpy as np

# local imports
import sys
sys.path.append(str(Path(__file__).parents[1]))
from utils.plotting import plot_posteriors_violin, plot_descriptive_adequacy
from utils.helper_functions import chance_level, inv_logit


if __name__ in "__main__":
    path = Path(__file__).parent

    # load posterior and behavioral data
    inpath = path / "fit" / "param_est_HC_AD.csv"
    est_data = pd.read_csv(inpath)

    # rename the columns
    est_data.rename(columns = {"delta.1": "delta_a_rew", "delta.2": "delta_a_pun", "delta.3": "delta_K", "delta.4": "delta_theta", "delta.5": "delta_omega_p", "delta.6": "delta_omega_f"}, inplace = True)

    for col in est_data.columns[:20]:
        print(col)

        # print the mean and the 95% HDI
        print(f"Mean: {np.mean(est_data[col])}")
        print(f"max: {np.max(est_data[col])}")
        print(f"min: {np.min(est_data[col])}")


    AD_data = pd.read_csv(path / "data" / "AD" / "data_AD_all_subjects.csv")
    HC_data = pd.read_csv(path / "data" / "HC" / "data_HC_all_subjects.csv")
    HC_data["sub"] += AD_data["sub"].max() 

    data = pd.concat([AD_data, HC_data])
    n_subjects = data["sub"].nunique()

    outpath = path / "fig"

    if not outpath.exists():
        outpath.mkdir(parents = True)


    # plot the posterior densities of the parameters
    parameters = ["delta_a_rew", "delta_a_pun", "delta_K", "delta_theta", "delta_omega_p", "delta_omega_f"]

    posteriors = [np.array(est_data[param]) for param in parameters]

    parameter_names = ["$\Delta A_{rew}$", "$\Delta A_{pun}$", "$\Delta K$", "$\Delta \\theta$", "$\Delta \omega_P$", "$\Delta \omega_F$"]

    plot_posteriors_violin(posteriors, parameter_names, savepath = outpath / "posterior_densities.png")

    pred_choices = []
    # get the predicted choices
    for sub in range(1, n_subjects+1):
        pred_choices_sub = []
        
        for t in range(1, 101):
            pred_choices_sub.append(mode(est_data[f"y_pred.{sub}.{t}"]))

        pred_choices.append(pred_choices_sub)

    # get the actual choices 
    choices = []
    for sub in data["sub"].unique():
        tmp_data = data[data["sub"] == sub]
        choices.append(tmp_data["choice"].to_list())

    # plot the descriptive adequacy of the model
    plot_descriptive_adequacy(
        choices, 
        pred_choices, 
        groups = [0] * AD_data["sub"].nunique()+ [1] * HC_data["sub"].nunique(),
        group_labels = {0: "AD", 1: "HC"},
        chance_level = chance_level(n = 100, p = 0.25, alpha = 0.5)*100,
        sort_accuracy = True,
        savepath = outpath / "descriptive_adequacy_sorted.png"
        )

    plot_descriptive_adequacy(
        choices, 
        pred_choices, 
        groups = [0] * AD_data["sub"].nunique()+ [1] * HC_data["sub"].nunique(),
        group_labels = {0: "AD", 1: "HC"},
        chance_level = chance_level(n = 100, p = 0.25, alpha = 0.5)*100,
        savepath = outpath / "descriptive_adequacy.png"
        )

    # print mean accuracy of each group
    for group in [0, 1]:
        group_inds = [ind for ind, g in enumerate([0] * AD_data["sub"].nunique()+ [1] * HC_data["sub"].nunique()) if g == group]
        group_mean = np.mean([percent_correct[ind] for ind in group_inds])
        print(f"Mean accuracy of group {group}: {group_mean}")