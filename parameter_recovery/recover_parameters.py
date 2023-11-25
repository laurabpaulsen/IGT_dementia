import stan
from generate import simulate_ORL
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def MPD(x: pd.Series):
    """
    Modified from Andreas' R code.
    
    # defining a function for calculating the maximum of the posterior density (not exactly the same as the mode)
    MPD <- function(x) {
    density(x)$x[which(density(x)$y==max(density(x)$y))]
    }
    """
    density = x.plot.density()
    return density.x[np.argmax(density.y)]

def plot_recovery_ax(ax, true, estimated, parameter_name):
    """
    Helper function for plot_recoveries
    """
    ax.scatter(true, estimated)
    x_lims = ax.get_xlim()
    ax.plot([0, x_lims[1]], [0, x_lims[1]], color = "black", linestyle = "dashed")
    ax.set_xlabel("True")
    ax.set_ylabel("Estimated")
    ax.set_title(parameter_name.title())


def plot_recoveries(trues:list, estimateds:list, parameter_names:list, savepath:Path):
    """
    Plot the recovery of the parameters.

    Parameters
    ----------
    trues : list
        List of true parameters.
    estimateds : list
        List of estimated parameters.
    parameter_names : list
        List of parameter names.
    savepath : Path
        Path to save the figure to.
    
    Returns
    -------
    None
    """

     # plot true vs estimated parameters
    fig, axes = plt.subplots(1, len(trues), figsize = (15, 5))
    
    for true, estimated, parameter_name, axis in zip(trues, estimateds, parameter_names, axes):
        plot_recovery_ax(axis, true, estimated, parameter_name)

    plt.tight_layout()
    
    if savepath:
        plt.savefig(savepath)



def test_parameter_recovery(n_subjects, model_spec, savepath = None):
    """
    Generate synthetic data and fit the model to it. Check how well the parameters are recovered by plotting median against the true parameters.

    Parameters
    ----------
    n_subjects : int
        Number of subjects.
    model_spec : str
        Stan model specification.
    savepath : Path, optional
        Path to save the parameter recovery figure to, by default None
    """

    # for storing true parameters
    a_rew_t, a_pun_t = np.zeros(n_subjects), np.zeros(n_subjects)

    # for storing estimated parameters
    a_rew_e, a_pun_e = np.zeros(n_subjects), np.zeros(n_subjects)

    for sub in range(n_subjects):
        a_pun = np.random.uniform(0, 1)
        a_rew = np.random.uniform(0, 1)
        # generate synthetic data
        data = simulate_ORL(
            a_rew = a_rew,
            a_pun = a_pun,
        )

        # fit the model
        model = stan.build(model_spec, data = data)
        fit = model.sample(num_chains = 4, num_samples = 1000)

        # get the estimated parameters
        estimated_parameters = fit.get_posterior_mean()

        # store the true and estimated parameters
        a_pun_t[sub], a_rew_t[sub] = a_pun, a_rew
        a_pun_e[sub], a_rew_e[sub] = estimated_parameters["a_pun"], estimated_parameters["a_rew"]

    
    # plot the recovery of the parameters
    plot_recoveries(
        trues = [a_pun_t, a_rew_t],
        estimateds = [a_pun_e, a_rew_e],
        parameter_names = ["a_pun", "a_rew"],
        savepath = savepath
    )


if __name__ == "__main__":
    path = Path(__file__).parent

    outpath = path / "fig"

    if not outpath.exists():
        outpath.mkdir()


    test_parameter_recovery(
        n_subjects = 100,
        model_spec = "models/ORL.stan",
        savepath = path / "fig" / "parameter_recovery_ORL.png"
    )