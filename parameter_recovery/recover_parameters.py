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


def plot_recoveries(trues:list, estimateds:list, parameter_names:list, savepath: Path):
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



def test_parameter_recovery(n_subjects, model_spec, savepath_fig = None, savepath_df = None):
    """
    Generate synthetic data and fit the model to it. Check how well the parameters are recovered by plotting median against the true parameters.

    Parameters
    ----------
    n_subjects : int
        Number of subjects.
    model_spec : str
        Stan model specification.
    savepath_fig : Path, optional
        Path to save the parameter recovery figure to, by default None
    savepath_df : Path
        Path to save the df to. NOT IMPLEMENTED CORRECTLY, currently overwrites for each participant
    """

    # for storing true parameters
    a_rew_t, a_pun_t, K_t, omega_f_t, omega_p_t = np.zeros(n_subjects), np.zeros(n_subjects), np.zeros(n_subjects), np.zeros(n_subjects), np.zeros(n_subjects)

    # for storing estimated parameters
    a_rew_e, a_pun_e, K_e, omega_f_e, omega_p_e = np.zeros(n_subjects), np.zeros(n_subjects), np.zeros(n_subjects), np.zeros(n_subjects), np.zeros(n_subjects)

    for sub in range(n_subjects):
        a_pun = np.random.uniform(0, 1)
        a_rew = np.random.uniform(0, 1)
        K = np.random.uniform(0, 1)
        omega_f = np.random.uniform(-2, 2)
        omega_p = np.random.uniform(-2, 2)

        # generate synthetic data
        data = simulate_ORL(
            a_rew = a_rew,
            a_pun = a_pun,
            K = K,
            omega_f = omega_f, 
            omega_p = omega_p,
            theta = 1

        )

        # fit the model
        model = stan.build(model_spec, data = data)
        fit = model.sample(num_chains = 4, num_samples = 1000)

        # get the estimated parameters
        df = fit.to_frame()
        
        if savepath_df:
            df.to_csv(savepath_df)

        # store the true and estimated parameters
        a_rew_t[sub], a_pun_t[sub], K_t[sub], omega_f_t[sub], omega_p_t[sub] = a_pun, a_rew, K, omega_f, omega_p
        a_rew_e[sub], a_pun_e[sub], K_e[sub], omega_f_e[sub], omega_p_e[sub] = df["a_pun"].mean(), df["a_rew"].mean(), df["K"].mean(), df["omega_f"].mean(), df["omega_p"].mean()

        # ADD THETA????
    
    # plot the recovery of the parameters
    plot_recoveries(
        trues = [a_rew_t, a_pun_t, K_t, omega_f_t, omega_p_t],
        estimateds = [a_rew_e, a_pun_e, K_e, omega_f_e, omega_p_e],
        parameter_names = ["a_pun", "a_rew", "K", "omega_f", "omega_p"],
        savepath = savepath_fig
    )


if __name__ == "__main__":
    path = Path(__file__).parent

    outpath = path / "fig"

    if not outpath.exists():
        outpath.mkdir()

    with open(path.parent / "single_subject.stan") as f:
        model_spec = f.read()

    test_parameter_recovery(
        n_subjects = 30,
        model_spec = model_spec,
        savepath_fig = path / "fig" / "subject_level_parameter_recovery_ORL.png",
        #savepath_df = path / "subject_level_parameter_recovery_ORL.csv"
    )