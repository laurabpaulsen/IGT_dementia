import pandas as pd
from pathlib import Path
from statistics import mode
import numpy as np

# local imports
import sys
sys.path.append(str(Path(__file__).parents[1]))
from utils.plotting import plot_posteriors_violin, plot_descriptive_adequacy
from utils.helper_functions import chance_level


if __name__ in "__main__":
    path = Path(__file__).parent

    # load posterior and behavioral data
    inpath = path / "fit" / "param_est_HC_AD.csv"
    est_data = pd.read_csv(inpath)

    AD_data = pd.read_csv(path / "data" / "AD" / "data_AD_all_subjects.csv")
    HC_data = pd.read_csv(path / "data" / "HC" / "data_HC_all_subjects.csv")
    HC_data["sub"] += AD_data["sub"].max() 

    data = pd.concat([AD_data, HC_data])
    n_subjects = data["sub"].nunique()

    outpath = path / "fig"

    if not outpath.exists():
        outpath.mkdir(parents = True)


    # plot the posterior densities of the parameters
    parameters = ["delta_a_rew", "delta_a_pun", "delta_K", "delta_omega_p", "delta_omega_f"]
    densities = [np.array(est_data[parameter]) for parameter in parameters]
    parameter_names = ["$\Delta A_{rew}$", "$\Delta A_{pun}$", "$\Delta K$", "$\Delta \omega_{p}$", "$\Delta \omega_{f}$"]

    plot_posteriors_violin(densities, parameter_names, savepath = outpath / "posterior_densities.png")

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