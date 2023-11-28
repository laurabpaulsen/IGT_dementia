"""
Plotting module for the project. Functions are used for both parameter recovery and parameter estimation.
"""
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


def plot_descriptive_adequacy(choices, pred_choices, groups, chance_level = None, savepath: Path = None):
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
    n_sub = len(choices)
    percent_correct = []
    
    for sub in range(n_sub):
        correct = [choice == pred_choice for choice, pred_choice in zip(choices[sub], pred_choices[sub])]
        sum_correct = sum(correct)
        sum_choices = len(choices[sub])
        percent_correct.append(sum_correct/sum_choices*100)

    # plot the accuracy
    fig, ax = plt.subplots(1, 1, figsize = (5, 5), dpi = 300)

    # plot the accuracy as bar plot but color the bars according to the group
    ax.bar(range(n_sub), percent_correct, color = [f"C{group-1}" for group in groups])

    # plot the chance level
    if chance_level:
        ax.axhline(chance_level, color = "black", linestyle = "dashed", label = "Chance level", linewidth = 0.5)
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