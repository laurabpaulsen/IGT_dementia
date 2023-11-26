import stan
from generate import simulate_ORL_group
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



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



def test_parameter_recovery_grouplevel(n_groups, n_subjects, model_spec, savepath_fig = None, savepath_df = None):
    """
    Generate synthetic data and fit the model to it. Check how well the parameters are recovered by plotting median against the true parameters.

    Parameters
    ----------
    n_groups : int
        Number of groups to simulate.
    model_spec : str
        Stan model specification.
    savepath_fig : Path, optional
        Path to save the parameter recovery figure to, by default None
    """

    mu_a_rew_t = np.zeros(n_groups)
    mu_a_pun_t = np.zeros(n_groups)
    mu_K_t = np.zeros(n_groups)
    mu_omega_f_t = np.zeros(n_groups)
    mu_omega_p_t = np.zeros(n_groups)

    mu_a_rew_e = np.zeros(n_groups)
    mu_a_pun_e = np.zeros(n_groups)
    mu_K_e = np.zeros(n_groups)
    mu_omega_f_e = np.zeros(n_groups)
    mu_omega_p_e = np.zeros(n_groups)



    for group in range(n_groups):
        mu_a_rew = np.random.uniform(0, 1)
        mu_a_pun = np.random.uniform(0, 1)
        mu_K = np.random.uniform(0, 2)
        mu_omega_f = np.random.uniform(0, 2)
        mu_omega_p = np.random.uniform(0, 2)

        data = simulate_ORL_group(
            n_subjects = n_subjects,
            mu_a_rew = mu_a_rew,
            mu_a_pun = mu_a_pun,
            mu_K = mu_K,
            mu_omega_f = mu_omega_f,
            mu_omega_p = mu_omega_p,
            )

        # fit the model
        model = stan.build(model_spec, data = data)
        fit = model.sample(num_chains = 4, num_samples = 1000)

        # get the estimated parameters
        df = fit.to_frame()

        mu_a_rew_e[group] = df["mu_a_rew"].median()
        mu_a_pun_e[group] = df["mu_a_pun"].median()
        mu_K_e[group] = df["mu_K"].median()
        mu_omega_f_e[group] = df["mu_omega_f"].median()
        mu_omega_p_e[group] = df["mu_omega_p"].median()

        # store the true parameters
        mu_a_rew_t[group] = mu_a_rew
        mu_a_pun_t[group] = mu_a_pun
        mu_K_t[group] = mu_K
        mu_omega_f_t[group] = mu_omega_f
        mu_omega_p_t[group] = mu_omega_p

    # plot the recovery of the parameters
    plot_recoveries(
        trues = [mu_a_rew_t, mu_a_pun_t, mu_K_t, mu_omega_f_t, mu_omega_p_t],
        estimateds = [mu_a_rew_e, mu_a_pun_e, mu_K_e, mu_omega_f_e, mu_omega_p_e],
        parameter_names = ["mu_a_pun", "mu_a_rew", "mu_K", "mu_omega_f", "mu_omega_p"],
        savepath = savepath_fig
    )


if __name__ == "__main__":
    path = Path(__file__).parent

    outpath = path / "fig"

    if not outpath.exists():
        outpath.mkdir()

    with open(path.parent / "hierachical_IGT_ORL.stan") as f:
        model_spec = f.read()

    test_parameter_recovery_grouplevel(
        n_groups = 50,
        n_subjects = 30,
        model_spec = model_spec,
        savepath_fig = path / "fig" / "hierachical_parameter_recovery_ORL.png",
        #savepath_df = path / "hierachical_parameter_recovery_ORL.csv"
    )