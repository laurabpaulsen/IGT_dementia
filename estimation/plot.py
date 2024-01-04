import pandas as pd
from pathlib import Path
from statistics import mode
import numpy as np

# local imports
import sys
sys.path.append(str(Path(__file__).parents[1]))
from utils.plotting import plot_posteriors_violin, plot_descriptive_adequacy, plot_posterior
from utils.helper_functions import chance_level


if __name__ in "__main__":
    path = Path(__file__).parent

    # load posterior and behavioral data
    inpath = path / "fit" / "param_est_HC_AD.csv"
    est_data = pd.read_csv(inpath)

    # rename the columns
    est_data.rename(columns = {"delta.1": "delta_a_rew", "delta.2": "delta_a_pun", "delta.3": "delta_K", "delta.4": "delta_omega_p", "delta.5": "delta_omega_f"}, inplace = True)


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

    posteriors = [np.array(est_data[param]) for param in parameters]

    # normal distributed priors
    prior1 = np.random.normal(0, 1, 1000)

    parameter_names = ["$\Delta A_{rew}$", "$\Delta A_{pun}$", "$\Delta K$", "$\Delta \omega_P$", "$\Delta \omega_F$"]
    priors = [prior1] * 5

    plot_posterior(priors, posteriors, parameter_names, savepath = outpath / "posterior_densities_priors.png")



    #plot_posteriors_violin(posteriors, parameter_names, savepath = outpath / "posterior_densities.png")
