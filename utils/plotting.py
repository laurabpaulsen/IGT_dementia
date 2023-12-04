"""
Plotting module for the project. Functions are used for both parameter recovery and parameter estimation.
"""
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

colours = ["steelblue", "lightblue"]

def plot_descriptive_adequacy(
    choices, 
    pred_choices, 
    groups = None, 
    group_labels:dict = None, 
    chance_level = None, 
    sort_accuracy = False,
    savepath: Path = None
    ):
    """
    Plot the descriptive adequacy of the model, that is, how well the model predicts the deck choice of the participants.

    Parameters
    ----------
    choices : list
        List of lists of choices.
    pred_choices : list
        List of lists of predicted choices.
    groups : list
        List of groups.
    chance_level : float, optional
        If provided a horizontal line is drawn at the chance level. If None, no line is drawn.
    savepath : Path = None
        Path to save the figure to. If None, the figure is not saved.
    """

    # Calculate the correct choices per participant
    n_sub = len(pred_choices)
    percent_correct = []
    
    for sub in range(n_sub):
        correct = [choice == pred_choice for choice, pred_choice in zip(choices[sub], pred_choices[sub])]
        sum_correct = sum(correct)
        sum_choices = len(choices[sub])
        percent_correct.append(sum_correct/sum_choices*100)

    if sort_accuracy:
        sort_inds = np.argsort(percent_correct)[::-1]
        percent_correct = [percent_correct[ind] for ind in sort_inds]

        if groups:
            groups = [groups[ind] for ind in sort_inds]


    # plot the accuracy
    fig, ax = plt.subplots(1, 1, figsize = (5, 5), dpi = 300)

    # plot the accuracy as bar plot but color the bars according to the group
    if groups:
        ax.bar(range(1, n_sub + 1), percent_correct, color = [colours[group] for group in groups])
    else:
        ax.bar(range(1, n_sub + 1), percent_correct)
    # plot the chance level
    
    if chance_level:
        ax.axhline(chance_level, color = "black", linestyle = "dashed", label = "Chance level", linewidth = 0.5)
    
    # add labels for legend
    if group_labels:
        for group in group_labels:
            ax.bar([0], [0], color = colours[group], label = group_labels[group])
    
    if chance_level or group_labels:
        ax.legend()

    ax.set_xlabel("Subject")
    ax.set_ylabel("Accuracy [%]")



    ax.set_xlim(1, n_sub+1)
    

    plt.tight_layout()

    if savepath:
        plt.savefig(savepath)



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
    fig, axes = plt.subplots(2, len(trues) // 2 + (len(trues) % 2 > 0), figsize = (10, 7), dpi = 300)
    
    for true, estimated, parameter_name, axis in zip(trues, estimateds, parameter_names, axes.flatten()):
        plot_recovery_ax(axis, true, estimated, parameter_name)

    # if any of the axes is empty, remove it
    for axis in axes.flatten():
        if not axis.get_title():
            fig.delaxes(axis)
    
    plt.tight_layout()
    
    if savepath:
        plt.savefig(savepath)

def plot_recovery_ax(ax, true, estimated, parameter_name):
    """
    Helper function for plot_recoveries
    """
    ax.scatter(true, estimated, s=10)
    x_lims = ax.get_xlim()
    y_lims = ax.get_ylim()
    ax.plot([y_lims[0], x_lims[1]], [y_lims[0], x_lims[1]], color = "black", linestyle = "dashed")
    ax.set_xlabel("True")
    ax.set_ylabel("Estimated")
    ax.set_title(parameter_name.title())



def plot_posteriors_violin(densities, parameter_names, savepath = None):
    """
    Plot the posterior densities of a set of parameters as violin plots.
    
    Parameters
    ----------
    densities : list of np.arrays
        List of posterior densities.
    parameter_names : list of str
        List of parameter names.
    savepath : Path, optional
        Path to save the figure to. The default is None.
    """
    fig, ax = plt.subplots(1, 1, figsize = (7, 5), dpi = 300)

    # line at 0
    ax.axhline(0, color = "black", linestyle = "dashed", linewidth = 0.5)

    ax.violinplot(densities, showextrema = False, widths=0.8)

    # plot quantiles as whiskers
    for i, density in enumerate(densities):
        quantiles = np.quantile(density, [0.025, 0.975])
        ax.plot([i+1, i+1], quantiles, color = "midnightblue", linewidth = 2)

    # plot median with a white dot
    medians = [np.median(density) for density in densities]
    ax.scatter(range(1, len(medians)+1), medians, color = "white", s = 15, zorder = 3)


    ax.set_xticks(range(1, len(parameter_names)+1))
    ax.set_xticklabels(parameter_names)

    # aesthetics to make the plot look nicer
    ax.set_ylabel("Posterior density")
    ax.set_xlabel("Parameter")

    plt.tight_layout()    

    if savepath:
        plt.savefig(savepath)


