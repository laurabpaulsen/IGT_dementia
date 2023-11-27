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