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



def test_parameter_recovery_grouplevel(n_subjects, model_spec, savepath_fig = None, savepath_df = None):
    """
    Generate synthetic data and fit the model to it. Check how well the parameters are recovered by plotting median against the true parameters.

    Parameters
    ----------
    n_subjects : int
        Number of subjects in the group
    model_spec : str
        Stan model specification.
    savepath_fig : Path, optional
        Path to save the parameter recovery figure to, by default None
    savepath_df : Path
        Path to save the df to. NOT IMPLEMENTED CORRECTLY, currently overwrites for each participant
    """

    mu_a_rew = 0.3
    mu_a_pun = 0.3
    mu_K = 0.3
    mu_omega_f = 0.3
    mu_omega_p = 0.3

    

    for sub in range(n_subjects):
        # generate parameters
        a_rew = np.random.normal(mu_a_rew, 0.1)
        a_pun = np.random.normal(mu_a_pun, 0.1)
        K = np.random.normal(mu_K, 0.1)
        omega_f = np.random.normal(mu_omega_f, 0.1)
        omega_p = np.random.normal(mu_omega_p, 0.1)

        if sub == 0:
            # generate synthetic data
            data = simulate_ORL(
                a_rew = a_rew,
                a_pun = a_pun,
                K = K,
                omega_f = omega_f, 
                omega_p = omega_p,
                theta = 1

            )
        else: 
            data_tmp = simulate_ORL(
                a_rew = a_rew,
                a_pun = a_pun,
                K = K,
                omega_f = omega_f, 
                omega_p = omega_p,
                theta = 1
            )

            # extend the data with the new subject
            for key in data_tmp.keys():
                data[key].extend(data_tmp[key])



    # fit the model
    model = stan.build(model_spec, data = data)
    fit = model.sample(num_chains = 4, num_samples = 1000)

    # get the estimated parameters
    df = fit.to_frame()
        
    if savepath_df:
        df.to_csv(savepath_df)

        # ADD THETA????
    
    # plot the recovery of the parameters
    plot_recoveries(
        trues = [mu_a_rew, mu_a_pun, mu_K, mu_omega_f, mu_omega_p],
        estimateds = [MPD(df["a_rew"]), MPD(df["a_pun"]), MPD(df["K"]), MPD(df["omega_f"]), MPD(df["omega_p"])],
        parameter_names = ["a_pun", "a_rew", "K", "omega_f", "omega_p"],
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
        n_subjects = 30,
        model_spec = model_spec,
        savepath_fig = path / "fig" / "hierachical_parameter_recovery_ORL.png",
        #savepath_df = path / "hierachical_parameter_recovery_ORL.csv"
    )