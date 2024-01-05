import pandas as pd
from pathlib import Path
from statistics import mode
import numpy as np

# local imports
import sys
sys.path.append(str(Path(__file__).parents[1]))
from utils.plotting import plot_posteriors_violin, plot_descriptive_adequacy, plot_posterior
from utils.helper_functions import chance_level

def credible_interval_table(posterior_dict):
    """
    Create a table with the 95% credible intervals for each parameter.
    """
    # get the 95% credible interval for each parameter
    credible_intervals = [np.quantile(posterior_dict[param], [0.025, 0.975]) for param in posterior_dict.keys()]

    # create a dataframe
    df = pd.DataFrame(credible_intervals, index = posterior_dict.keys(), columns = ["lower", "upper"])

    return df

if __name__ in "__main__":
    path = Path(__file__).parent

    outpath = path / "fig"
    resultspath = path / "results"

    if not outpath.exists():
        outpath.mkdir(parents = True)
    
    if not resultspath.exists():
        resultspath.mkdir(parents = True)

    parameters = ["delta_a_rew", "delta_a_pun", "delta_K", "delta_omega_p", "delta_omega_f"]

    
    for file_end, posterior_path in zip(["_1", "_2", "_pooled"], ["param_est_HC_AD_1.csv", "param_est_HC_AD_2.csv", "param_est_HC_AD_pooled.csv"]):
        inpath = path / "fit" / posterior_path
        figpath = outpath / f"posterior_densities{file_end}.png" 

        est_data = pd.read_csv(inpath)

        # rename the columns
        est_data.rename(columns = {"delta.1": "delta_a_rew", "delta.2": "delta_a_pun", "delta.3": "delta_K", "delta.4": "delta_omega_p", "delta.5": "delta_omega_f"}, inplace = True)

        posteriors = [np.array(est_data[param]) for param in parameters]

        # normal distributed priors
        prior1 = np.random.normal(0, 1, 1000)

        parameter_names = ["$\Delta A_{rew}$", "$\Delta A_{pun}$", "$\Delta K$", "$\Delta \omega_P$", "$\Delta \omega_F$"]
        priors = [prior1] * 5

        plot_posterior(
            priors, 
            posteriors, 
            parameter_names, 
            savepath = figpath
        )

        # make a dicitonary with the parameters and the posteriors
        posterior_dict = dict(zip(parameters, posteriors))

        # create a table with the credible intervals
        table = credible_interval_table(posterior_dict)
        table = table.round(2)

        table.to_csv(resultspath / f"credible_intervals{file_end}.csv")
