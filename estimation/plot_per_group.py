import pandas as pd
from pathlib import Path
from statistics import mode
import numpy as np

# local imports
import sys
sys.path.append(str(Path(__file__).parents[1]))
from utils.plotting import plot_posteriors_violin, plot_descriptive_adequacy, plot_compare_posteriors, plot_compare_prior_posteriors
from utils.helper_functions import chance_level, inv_logit


if __name__ in "__main__":
    path = Path(__file__).parent

    # load posterior and behavioral data
    inpath_ad = path / "fit" / "param_est_AD.csv"
    inpath_hc = path / "fit" / "param_est_HC.csv"
    
    est_data_ad = pd.read_csv(inpath_ad)
    est_data_hc = pd.read_csv(inpath_hc)

    # rename the columns
    est_data_ad.rename(columns = {"mu.1": "mu_a_rew", "mu.2": "mu_a_pun", "mu.3": "mu_K", "mu.4": "mu_theta", "mu.5": "mu_omega_p", "mu.6": "mu_omega_f"}, inplace = True)
    est_data_hc.rename(columns = {"mu.1": "mu_a_rew", "mu.2": "mu_a_pun", "mu.3": "mu_K", "mu.4": "mu_theta", "mu.5": "mu_omega_p", "mu.6": "mu_omega_f"}, inplace = True)

    AD_data = pd.read_csv(path / "data" / "AD" / "data_AD_all_subjects.csv")
    HC_data = pd.read_csv(path / "data" / "HC" / "data_HC_all_subjects.csv")
    AD_data["group"] = "AD"
    HC_data["group"] = "HC"

    HC_data["sub"] += AD_data["sub"].max() 
    data = pd.concat([AD_data, HC_data])

    #n_subjects = data["sub"].nunique()

    outpath = path / "fig"

    if not outpath.exists():
        outpath.mkdir(parents = True)

    # plot the posterior densities of the parameters
    parameters = ["mu_a_rew", "mu_a_pun", "mu_K", "mu_theta", "mu_omega_p", "mu_omega_f"]

    posteriors_ad = [np.array(est_data_ad[param]) for param in parameters]
    posteriors_hc = [np.array(est_data_hc[param]) for param in parameters]

    parameter_names = ["$\mu A_{rew}$", "$\mu A_{pun}$", "$\mu K$", "$\mu \\theta$", "$\mu \omega_P$", "$\mu \omega_F$"]

    plot_posteriors_violin(posteriors_ad, parameter_names, savepath = outpath / "posterior_densities_AD.png")
    plot_posteriors_violin(posteriors_hc, parameter_names, savepath = outpath / "posterior_densities_HC.png")

    pred_choices = []
    
    # not 100 procent sure that this order alligns with the order of the subjects in the data
    for group, est_data, n_sub in zip(["AD", "HC"], [est_data_ad, est_data_hc], [AD_data["sub"].nunique(), HC_data["sub"].nunique()]):
        for sub in range(1, n_sub + 1):
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
        savepath = outpath / "descriptive_adequacy_sorted_per_group.png"
        )

    plot_descriptive_adequacy(
        choices, 
        pred_choices, 
        groups = [0] * AD_data["sub"].nunique()+ [1] * HC_data["sub"].nunique(),
        group_labels = {0: "AD", 1: "HC"},
        chance_level = chance_level(n = 100, p = 0.25, alpha = 0.5)*100,
        savepath = outpath / "descriptive_adequacy_per_group.png"
        )

    plot_compare_posteriors(
        posteriors_ad, 
        posteriors_hc, 
        parameter_names, 
        group_labels = ["AD", "HC"],
        savepath = outpath / "compare_posteriors.png"
        )

    # plot the prior and posterior densities
    priors = []

    uniform = np.random.uniform(0, 1, 10000)
    priors.append(uniform)
    priors.append(uniform)

    normal_truncated = np.random.normal(0, 1, 10000)
    normal_truncated[normal_truncated < 0] = 0
    priors.append(normal_truncated)
    priors.append(normal_truncated)

    normal = np.random.normal(0, 1, 10000)
    priors.append(normal)
    priors.append(normal)

    plot_compare_prior_posteriors(
        priors, 
        posteriors_ad, 
        posteriors_hc,
        parameter_names, 
        group_labels = ["AD", "HC"],
        savepath = outpath / "compare_prior_posteriors.png"
        )

