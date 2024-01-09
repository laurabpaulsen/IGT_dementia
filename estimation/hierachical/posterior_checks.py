import pandas as pd
from pathlib import Path
from statistics import mode
import numpy as np
import matplotlib.pyplot as plt

# local imports
import sys
sys.path.append(str(Path(__file__).parents[2]))
sys.path.append(str(Path(__file__).parents[2] / "estimation" / "hierachical_compare"))
from utils.plotting import plot_descriptive_adequacy, plot_traceplots, plot_trankplot
from utils.helper_functions import chance_level
from estimate_pooled import load_behavioural_pooled


def investigate_descriptive_adquacy(data, est_data):
    
    pred_choices = []
    # get the predicted choices
    for sub in range(1, data["sub"].nunique() + 1):
        pred_choices_sub = []
        
        for t in range(1, 101):
            pred_choices_sub.append(mode(est_data[f"y_pred.{sub}.{t}"]))

        pred_choices.append(pred_choices_sub)


    # get the actual choices 
    choices = []
    for sub in data["sub"].unique():
        tmp_data = data[data["sub"] == sub]
        choices.append(tmp_data["choice"].to_list())

    return choices, pred_choices



def load_HC(path):
    est_data = pd.read_csv(path / "fit" / "param_est_HC_pooled.csv")
    est_data.rename(columns = {"mu.1": "mu_a_rew", "mu.2": "mu_a_pun", "mu.3": "mu_K", "mu.4": "mu_omega_p", "mu.5": "mu_omega_f"}, inplace = True)

    summary_stats = pd.read_csv(path / "fit" / "param_est_HC_summary_pooled.csv")
    summary_stats.rename(columns = {"Unnamed: 0": "parameter"}, inplace = True)

    return est_data, summary_stats
    

def load_AD(path):
    est_data = pd.read_csv(path / "fit" / "param_est_AD_pooled.csv")
    est_data.rename(columns = {"mu.1": "mu_a_rew", "mu.2": "mu_a_pun", "mu.3": "mu_K", "mu.4": "mu_omega_p", "mu.5": "mu_omega_f"}, inplace = True)
    
    summary_stats = pd.read_csv(path / "fit" / "param_est_AD_summary_pooled.csv")
    summary_stats.rename(columns = {"Unnamed: 0": "parameter"}, inplace = True)

    return est_data, summary_stats


if __name__ == "__main__":
    path = Path(__file__).parent

    outpath = path / "fig"
    if not outpath.exists():
        outpath.mkdir(parents = True)

    # plot traceplots of the parameters
    parameters = ["mu_a_rew", "mu_a_pun", "mu_K", "mu_omega_p", "mu_omega_f"]
    parameter_names = [r"$\mu A_{rew}$", r"$\mu A_{pun}$", r"$\mu K$", r"$\mu \omega_{p}$", r"$\mu \omega_{f}$"]

    AD_data, HC_data = load_behavioural_pooled(path.parent)

    for file_ending, load_func, data in zip(["_HC", "_AD"],[load_HC, load_AD], [HC_data, AD_data]):
        est_data, summary_stats = load_func(path)

        # get r_hat values of all parameters starting with mu
        r_hat = summary_stats[summary_stats["parameter"].str.startswith("mu")]["r_hat"].to_list()
        print(f"r_hat values of mu parameters: {r_hat}")

        # convert the data to the correct format
        traceplot_data = est_data[parameters]
        traceplot_data = traceplot_data.to_dict(orient = "list")
        # split each key into 4 list (one for each chain)
        traceplot_data = {key: [traceplot_data[key][i::4] for i in range(4)] for key in traceplot_data.keys()}

        plot_traceplots(traceplot_data, parameter_names, savepath = outpath / f"traceplot{file_ending}.png")

        plot_trankplot(traceplot_data, parameter_names, savepath = outpath / f"rankplot{file_ending}.png")

        if file_ending == "_HC":
            choices_hc, pred_choices_hc = investigate_descriptive_adquacy(HC_data, est_data)
        else:
            choices_ad, pred_choices_ad = investigate_descriptive_adquacy(AD_data, est_data)


    plot_descriptive_adequacy(
        choices_ad,
        choices_hc,
        pred_choices_ad,
        pred_choices_hc,
        group_labels = ["AD", "HC"],
        chance_level = chance_level(n = 100, p = 0.25, alpha = 0.5)*100,
        savepath = outpath / "descriptive_adequacy.png"
        )
    