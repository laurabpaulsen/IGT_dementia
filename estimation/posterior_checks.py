import pandas as pd
from pathlib import Path
from statistics import mode
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

# local imports
import sys
sys.path.append(str(Path(__file__).parents[1]))
from utils.plotting import plot_descriptive_adequacy
from utils.helper_functions import chance_level


def investigate_descriptive_adquacy(AD_data, HC_data, est_data, outpath):
    
    pred_choices_ad = []
    pred_choices_hc = []
    # get the predicted choices
    for sub in range(1, AD_data["sub"].nunique() + HC_data["sub"].nunique() + 1):
        pred_choices_sub = []
        
        for t in range(1, 101):
            pred_choices_sub.append(mode(est_data[f"y_pred.{sub}.{t}"]))

        if sub in AD_data["sub"].unique():
            pred_choices_ad.append(pred_choices_sub)
        else:
            pred_choices_hc.append(pred_choices_sub)

    # get the actual choices 
    choices_ad = []
    for sub in AD_data["sub"].unique():
        tmp_data = AD_data[AD_data["sub"] == sub]
        choices_ad.append(tmp_data["choice"].to_list())
    
    choices_hc = []
    for sub in HC_data["sub"].unique():
        tmp_data = HC_data[HC_data["sub"] == sub]
        choices_hc.append(tmp_data["choice"].to_list())

    plot_descriptive_adequacy(
        choices_ad,
        choices_hc,
        pred_choices_ad,
        pred_choices_hc,
        group_labels = ["AD", "HC"],
        chance_level = chance_level(n = 100, p = 0.25, alpha = 0.5)*100,
        savepath = outpath / "descriptive_adequacy.png"
        )



if __name__ == "__main__":
    path = Path(__file__).parent

    # load posterior, behavioural and summary data
    est_data = pd.read_csv(path / "fit" / "param_est_HC_AD.csv")
    est_data.rename(columns = {"delta.1": "delta_a_rew", "delta.2": "delta_a_pun", "delta.3": "delta_K", "delta.4": "delta_omega_p", "delta.5": "delta_omega_f"}, inplace = True)


    summary_stats = pd.read_csv(path / "fit" / "param_est_HC_AD_summary.csv")
    summary_stats.rename(columns = {"Unnamed: 0": "parameter"}, inplace = True)

    AD_data = pd.read_csv(path / "data" / "AD" / "data_AD_all_subjects.csv")
    HC_data = pd.read_csv(path / "data" / "HC" / "data_HC_all_subjects.csv")
    HC_data["sub"] += AD_data["sub"].max() 

    outpath = path / "fig"
    if not outpath.exists():
        outpath.mkdir(parents = True)


    # get r_hat values of all parameters starting with delta
    r_hat = summary_stats[summary_stats["parameter"].str.startswith("delta")]["r_hat"].to_list()
    print(f"r_hat values of delta parameters: {r_hat}")

    # plot traceplots of the parameters
    parameters = ["delta_a_rew", "delta_a_pun", "delta_K", "delta_omega_p", "delta_omega_f"]

    # convert the data to the correct format
    traceplot_data = est_data[parameters]
    traceplot_data = traceplot_data.to_dict(orient = "list")
    # split each key into 4 list (one for each chain)
    traceplot_data = {key: [est_data[key][i::4] for i in range(4)] for key in est_data.keys()}

    backend_kwargs = {"dpi" : 300, "figsize" : (12, 10)}

    az.plot_trace(traceplot_data, var_names = parameters, backend_kwargs = backend_kwargs)
    
    plt.savefig(outpath / f"traceplot.png")

    investigate_descriptive_adquacy(AD_data, HC_data, est_data, outpath)

    