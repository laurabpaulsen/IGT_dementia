import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

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



"""
    # plot the recovery of the parameters
    plot_recoveries(
        trues = [mu_a_rew_t, mu_a_pun_t, mu_K_t, mu_omega_f_t, mu_omega_p_t],
        estimateds = [mu_a_rew_e, mu_a_pun_e, mu_K_e, mu_omega_f_e, mu_omega_p_e],
        parameter_names = ["a_pun", "a_rew", "K", "omega_f", "omega_p"],
        savepath = savepath_fig / "hierachical_parameter_recovery_ORL.png"
    )
"""

if __name__ == "__main__":
    path = Path(__file__).parent

    fig_path = path / "fig"

    # create the figure path if it does not exist
    if not fig_path.exists():
        fig_path.mkdir()

    # load the simulated data
    filename = "ORL_simulated_10_groups_20_sub.csv"
    data = pd.read_csv(path / "simulated" / filename)

    # load the estimated data
    filename_est = f"param_rec_{filename}"
    data_estimated = pd.read_csv(path / "fit" / filename_est)

    # get the true parameters
    
    unique_groups = data["group"].unique()

    mu_a_rew_t = []
    mu_a_pun_t = []
    mu_K_t = []
    mu_omega_f_t = []
    mu_omega_p_t = []

    mu_a_rew_e = []
    mu_a_pun_e = []
    mu_K_e = []
    mu_omega_f_e = []
    mu_omega_p_e = []


    for group in unique_groups:
        data_tmp = data[data["group"] == group]
        mu_a_rew_t.append(data_tmp["mu_a_rew"].unique()[0])
        mu_a_pun_t.append(data_tmp["mu_a_pun"].unique()[0])
        mu_K_t.append(data_tmp["mu_K"].unique()[0])
        mu_omega_f_t.append(data_tmp["mu_omega_f"].unique()[0])
        mu_omega_p_t.append(data_tmp["mu_omega_p"].unique()[0])

        data_tmp = data_estimated[data_estimated["group"] == group]
        mu_a_rew_e.append(data_tmp["mu_a_rew"].mean())
        mu_a_pun_e.append(data_tmp["mu_a_pun"].mean())
        mu_K_e.append(data_tmp["mu_K"].mean())
        mu_omega_f_e.append(data_tmp["mu_omega_f"].mean())
        mu_omega_p_e.append(data_tmp["mu_omega_p"].mean())


    # plot the recovery of the parameters
    plot_recoveries(
        trues = [mu_a_rew_t, mu_a_pun_t, mu_K_t, mu_omega_f_t, mu_omega_p_t],
        estimateds = [mu_a_rew_e, mu_a_pun_e, mu_K_e, mu_omega_f_e, mu_omega_p_e],
        parameter_names = ["a_pun", "a_rew", "K", "omega_f", "omega_p"],
        savepath = fig_path / "hierachical_parameter_recovery_ORL.png"
    )


    # individual parameters
    # get the true parameters
    a_rew_t = []
    a_pun_t = []
    K_t = []
    omega_f_t = []
    omega_p_t = []
    
    a_rew_e = []
    a_pun_e = []
    K_e = []
    omega_f_e = []
    omega_p_e = []

    for group in unique_groups:
        data_tmp = data[data["group"] == group]
        data_tmp_est = data_estimated[data_estimated["group"] == group]

        # unique subject numbers
        unique_subjects = data_tmp["sub"].unique()

        for sub in unique_subjects:
            data_tmp_sub = data_tmp[data_tmp["sub"] == sub]
            a_rew_t.append(data_tmp_sub["sub_a_rew"].unique()[0])
            a_pun_t.append(data_tmp_sub["sub_a_pun"].unique()[0])
            K_t.append(data_tmp_sub["sub_K"].unique()[0])
            omega_f_t.append(data_tmp_sub["sub_omega_f"].unique()[0])
            omega_p_t.append(data_tmp_sub["sub_omega_p"].unique()[0])
            
            a_rew_e.append(data_tmp_est[f"a_rew.{sub}"].mean())
            a_pun_e.append(data_tmp_est[f"a_pun.{sub}"].mean())
            K_e.append(data_tmp_est[f"K.{sub}"].mean())
            omega_f_e.append(data_tmp_est[f"omega_f.{sub}"].mean())
            omega_p_e.append(data_tmp_est[f"omega_p.{sub}"].mean())
    
    # plot the recovery of the parameters
    plot_recoveries(
        trues = [a_rew_t, a_pun_t, K_t, omega_f_t, omega_p_t],
        estimateds = [a_rew_e, a_pun_e, K_e, omega_f_e, omega_p_e],
        parameter_names = ["a_pun", "a_rew", "K", "omega_f", "omega_p"],
        savepath = fig_path / "hierachical_parameter_recovery_ORL_individual.png"
    )