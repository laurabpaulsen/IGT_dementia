import pandas as pd
from pathlib import Path
from statistics import mode
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

from scipy.stats import norm

# local imports
import sys
sys.path.append(str(Path(__file__).parents[1]))
from utils.plotting import plot_posteriors_violin, plot_descriptive_adequacy
from utils.helper_functions import chance_level, inv_logit




def plot_posterior_ax(ax, posterior1, posterior2, parameter_name):
 
    credible_interval1 = np.quantile(posterior1, [0.025, 0.975])
    credible_interval2 = np.quantile(posterior2, [0.025, 0.975])

    print(f"{parameter_name}: {credible_interval1}")
    print(f"{parameter_name}: {credible_interval2}")
    print("-------------------")
    
    # only plot the credible interval for the first posterior
    sns.kdeplot(posterior1, ax = ax, color = "steelblue", fill = True, label = "Posterior", clip = (credible_interval1[0], credible_interval1[1]), alpha = 0.4, linewidth = 0.00001)

    # plot posterior with different colors for the credible interval
    sns.kdeplot(posterior1, ax = ax, color = "steelblue", fill = False, label = "Posterior", alpha = 1, linewidth = 2)
    

    # only plot the credible interval for the second posterior
    sns.kdeplot(posterior2, ax = ax, color = "forestgreen", fill = True, label = "Posterior", clip = (credible_interval2[0], credible_interval2[1]), alpha = 0.4, linewidth = 0.00001)

    # plot posterior with different colors for the credible interval
    sns.kdeplot(posterior2, ax = ax, color = "forestgreen", fill = False, label = "Posterior", alpha = 1, linewidth = 2)


    ax.set_title(parameter_name)

def plot_posterior(posterior1:list[list[float]], posterior2:list[list[float]], parameter_names:[list[str]], group_names:[list[str]], savepath:Path = None):

    fig, axes = plt.subplots(3, 2, figsize = (9, 10), dpi = 300)

    for pos1, pos2, ax, param in zip(posterior1, posterior2, axes.flatten(), parameter_names):
        plot_posterior_ax(ax, pos1, pos2, param)


    # dashed line for the prior, solid line for the posterior, and fill the area in between 
    custom_lines =[ plt.Line2D([0], [0], color = "steelblue", lw = 2),
                    Patch(facecolor = "steelblue", alpha = 0.4),
                    plt.Line2D([0], [0], color = "forestgreen", lw = 2),
                    Patch(facecolor = "forestgreen", alpha = 0.4)]

    axes[0, 0].legend(custom_lines, ["HC", "CI (95%)", "AD", "CI (95%)"], loc = "upper right")

    # if there is an empty axis, remove it
    for ax in axes.flatten():
        if not ax.get_title():
            fig.delaxes(ax)

    plt.tight_layout()

    if savepath:
        plt.savefig(savepath)



if __name__ in "__main__":
    path = Path(__file__).parent

    outpath = path / "fig"
    resultspath = path / "results"

    if not outpath.exists():
        outpath.mkdir(parents = True)
    
    if not resultspath.exists():
        resultspath.mkdir(parents = True)

    parameters = ["mu_a_rew", "mu_a_pun", "mu_K", "mu_omega_p", "mu_omega_f"]
    
    # normal distributed priors
    prior = np.random.normal(0, 1, 4000)

    # load the data
    HC = pd.read_csv(path / "fit"  / "param_est_HC_pooled.csv")
    AD = pd.read_csv(path / "fit"  / "param_est_AD_pooled.csv")
    
    HC.rename(columns = {"mu.1": "mu_a_rew", "mu.2": "mu_a_pun", "mu.3": "mu_K", "mu.4": "mu_omega_p", "mu.5": "mu_omega_f"}, inplace = True)
    AD.rename(columns = {"mu.1": "mu_a_rew", "mu.2": "mu_a_pun", "mu.3": "mu_K", "mu.4": "mu_omega_p", "mu.5": "mu_omega_f"}, inplace = True)

    HC_posteriors = [np.array(HC[param]) for param in parameters]
    AD_posteriors = [np.array(AD[param]) for param in parameters]

    HC_posteriors[:3] = [norm.cdf(x) for x in HC_posteriors[:3]]
    HC_posteriors[2] = HC_posteriors[2]*5

    AD_posteriors[:3] = [norm.cdf(x) for x in AD_posteriors[:3]]
    AD_posteriors[2] = AD_posteriors[2]*5


    parameter_names = ["$\mu A_{rew}$", "$\mu A_{pun}$", "$\mu K$", "$\mu \omega_P$", "$\mu \omega_F$"]


    plot_posterior(
            HC_posteriors, 
            AD_posteriors, 
            parameter_names,
            group_names = ["HC", "AD"], 
            savepath = outpath / f"posterior_densities_seperate.png"
    )

   
    """
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
    """