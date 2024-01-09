import pandas as pd
from pathlib import Path
from statistics import mode
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm

# local imports
import sys
sys.path.append(str(Path(__file__).parents[2]))
from utils.plotting import plot_posteriors_violin, plot_descriptive_adequacy, plot_posterior, plot_posterior_ax
from utils.helper_functions import chance_level, inv_logit

# import for kernel density estimation
from scipy.stats import gaussian_kde

def credible_interval_table(posterior_dict):
    """
    Create a table with the 95% credible intervals for each parameter.
    """
    # get the 95% credible interval for each parameter
    credible_intervals = [np.quantile(posterior_dict[param], [0.025, 0.975]) for param in posterior_dict.keys()]

    # get maximum likelihood estimates
    mle = [mode(posterior_dict[param]) for param in posterior_dict.keys()]


    # create a dataframe with the credible intervals and the MLE
    df = pd.DataFrame.from_dict({"param" : posterior_dict.keys(), "CI_low" : [x[0] for x in credible_intervals], "CI_high" : [x[1] for x in credible_intervals], "MLE" : mle})

    return df

def bayes_factor_table(posterior_dict, prior_dict):
    """
    Create a table with the Bayes factors for each parameter.
    """

    # get the Byes factors for each parameter about our belief that the parameter is different from 0
    
    data = pd.DataFrame()

    for posterior, prior, param in zip(posterior_dict.values(), prior_dict.values(), posterior_dict.keys()):
        # get the density at 0 for the prior
        prior_density = gaussian_kde(prior)
        prior_density_at_0 = prior_density(0)

        # get the density at 0 for the posterior
        posterior_density = gaussian_kde(posterior)
        posterior_density_at_0 = posterior_density(0)

        # calculate the Bayes factor
        bayes_factor = prior_density_at_0 / posterior_density_at_0

        data_dict = {
            "param" : param,
            "dens_0_posterior" : posterior_density_at_0,
            "dens_0_prior" : prior_density_at_0,
            "bayes_factor" : bayes_factor
        }

        tmp = pd.DataFrame.from_dict(data_dict)
        data = pd.concat([data, tmp], axis = 0)


    return data


if __name__ in "__main__":
    path = Path(__file__).parent

    outpath = path / "fig"
    resultspath = path / "results"

    if not outpath.exists():
        outpath.mkdir(parents = True)
    
    if not resultspath.exists():
        resultspath.mkdir(parents = True)

    parameters = ["delta_a_rew", "delta_a_pun", "delta_K", "delta_omega_p", "delta_omega_f"]
    
    # normal distributed priors
    prior = np.random.normal(0, 1, 4000)
    
    for file_end, posterior_path in zip(["_1", "_2", "_pooled"], ["param_est_HC_AD_1.csv", "param_est_HC_AD_2.csv", "param_est_HC_AD_pooled.csv"]):
        inpath = path / "fit" / posterior_path
        figpath = outpath / f"posterior_densities{file_end}.png" 

        est_data = pd.read_csv(inpath)

        # rename the columns
        est_data.rename(columns = {"delta.1": "delta_a_rew", "delta.2": "delta_a_pun", "delta.3": "delta_K", "delta.4": "delta_omega_p", "delta.5": "delta_omega_f"}, inplace = True)

        posteriors = [np.array(est_data[param]) for param in parameters]

        parameter_names = ["$\Delta A_{rew}$", "$\Delta A_{pun}$", "$\Delta K$", "$\Delta \omega_P$", "$\Delta \omega_F$"]
        priors = [prior] * 5

        plot_posterior(
            priors, 
            posteriors, 
            parameter_names, 
            savepath = figpath
        )

        # transform the posteriors to the original scale
        #posteriors_inv = [norm.cdf(x) for x in posteriors[:3]]
        #posteriors_inv[2] = posteriors_inv[2] *5

        #posteriors_rescaled = [(x-0.5)*2 for x in posteriors_inv]
        
        fig, axes = plt.subplots(1, 3, figsize = (10, 5))

        for ax, posterior, parameter_name in zip(axes, posteriors, parameter_names):
            plot_posterior_ax(ax, prior, posterior, parameter_name)
    
        for ax in axes:
            ax.set_xlabel("Parameter value")
            ax.set_ylabel("Density")
            ax.set_xlim(-1, 1)

        savepath = outpath / f"posterior_densities{file_end}_scaled.png" 
        
        plt.savefig(savepath, dpi = 300, bbox_inches = "tight")

        # make a dicitonary with the parameters and the posteriors
        posterior_dict = dict(zip(parameters, posteriors))
        prior_dict = dict(zip(parameters, priors))

        # create a table with the credible intervals
        table = credible_interval_table(posterior_dict)
        table = table.round(2)

        table.to_csv(resultspath / f"credible_intervals{file_end}.csv", index = False)

        # create a table with the Bayes factors
        table = bayes_factor_table(posterior_dict, prior_dict)
        #table = table.round(2)

        table.to_csv(resultspath / f"bayes_factors{file_end}.csv", index = False)
