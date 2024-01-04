import pandas as pd
from pathlib import Path
from statistics import mode
import numpy as np
import matplotlib.pyplot as plt

# local imports
import sys
sys.path.append(str(Path(__file__).parents[1]))
from utils.plotting import plot_descriptive_adequacy
from utils.helper_functions import chance_level

#colours = ["#588c7e", "#f2e394", "steelblue", "#d96459"]
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



def plot_traceplots(data, parameter_names = None, savepath = None):

    n_keys = len(data.keys())
    n_chains = len(data[list(data.keys())[0]])
    
    fig, axes = plt.subplots(n_keys, 1, figsize = (10, 12), dpi = 300)

    for i, key in enumerate(data.keys()):
        for chain in range(n_chains):
            axes[i].plot(data[key][chain], label = f"chain {chain+1}", linewidth = 0.5,  alpha = 1)
            
            # set x limits  
            axes[i].set_xlim(0, len(data[key][chain]))
        
        
        if parameter_names:
            axes[i].set_title(parameter_names[i])
        else:
            axes[i].set_title(key)
    
    plt.tight_layout()
    plt.legend()
    if savepath:
        plt.savefig(savepath)

def plot_trankplot(data, parameter_names = None, savepath = None):
    n_keys = len(data.keys())
    n_chains = len(data[list(data.keys())[0]])
    
    fig, axes = plt.subplots(n_keys, 1, figsize = (10, 12), dpi = 300)

    # join the chains for each parameter to get rankplot
    for i, key in enumerate(data.keys()):
        tmp_data = np.concatenate(data[key])

        # get the indices of the sorted array
        ranks = np.argsort(tmp_data)

        # get the ranks for each chain
        tmp_data = [ranks[i::n_chains] for i in range(n_chains)]

        step_size = 10
        print(len(tmp_data[0]))
        # get the count of within each step
        tmp_data = [np.array([np.sum((i <= tmp_data[chain]) & (tmp_data[chain] < i+step_size)) for i in range(0, len(tmp_data[chain]), step_size)]) for chain in range(n_chains)]

    
        # plot the rankplot
        for chain in range(n_chains):
            axes[i].step(np.arange(0, len(tmp_data[chain])),tmp_data[chain], label = f"chain {chain+1}", linewidth = 1,  alpha = 1)
            
            # set x limits  
            axes[i].set_xlim(0, len(tmp_data[chain]))

        if parameter_names:
            axes[i].set_title(parameter_names[i])
        else:
            axes[i].set_title(key)

    for ax in axes:
        # set the ticks to none 
        ax.set_xticks([])
        ax.set_yticks([])

        # get y limits
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(-1, y_max)

    plt.tight_layout()

    if savepath:
        plt.savefig(savepath)

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
    parameter_names = [r"$\Delta A_{rew}$", r"$\Delta A_{pun}$", r"$\Delta K$", r"$\Delta \omega_{p}$", r"$\Delta \omega_{f}$"]


    # convert the data to the correct format
    traceplot_data = est_data[parameters]
    traceplot_data = traceplot_data.to_dict(orient = "list")
    # split each key into 4 list (one for each chain)
    traceplot_data = {key: [traceplot_data[key][i::4] for i in range(4)] for key in traceplot_data.keys()}

    plot_traceplots(traceplot_data, parameter_names, savepath = outpath / "traceplot.png")

    plot_trankplot(traceplot_data, parameter_names, savepath = outpath / "rankplot.png")


    #investigate_descriptive_adquacy(AD_data, HC_data, est_data, outpath)

    