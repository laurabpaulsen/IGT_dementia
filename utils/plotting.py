"""
Plotting module for the project. Functions are used for both parameter recovery and parameter estimation.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from pathlib import Path
import numpy as np

# local imports
import sys
sys.path.append(str(Path(__file__).parent))
from helper_functions import maximum_posterior_density

colours = ["steelblue", "lightblue"]

def plot_descriptive_adequacy(
    group1_choices,
    group2_choices,
    group1_pred_choices,
    group2_pred_choices, 
    groups = None, 
    group_labels:dict = None, 
    chance_level = None, 
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
    percent_correct_group1 = []
    percent_correct_group2 = []
    
    for choices, pred_choices in zip(group1_choices, group1_pred_choices):
        percent_correct_group1.append(sum([choice == pred_choice for choice, pred_choice in zip(choices, pred_choices)]) / len(choices))
    
    for choices, pred_choices in zip(group2_choices, group2_pred_choices):
        percent_correct_group2.append(sum([choice == pred_choice for choice, pred_choice in zip(choices, pred_choices)]) / len(choices))
    

    percent_correct_group1 = [percent_correct*100 for percent_correct in percent_correct_group1]
    percent_correct_group2 = [percent_correct*100 for percent_correct in percent_correct_group2]

    fig, axes = plt.subplots(1, 2, figsize = (7, 5), dpi = 300)

    # plot the descriptive adequacy
    axes[0].bar(range(len(percent_correct_group1)), percent_correct_group1, color = colours[0], label = group_labels[0])
    axes[0].axhline(np.mean(percent_correct_group1), color = "black", linestyle = "solid", label = "Mean accuracy", linewidth = 1)

    axes[1].bar(range(len(percent_correct_group2)), percent_correct_group2, color = colours[1], label = group_labels[1])
    axes[1].axhline(np.mean(percent_correct_group2), color = "black", linestyle = "solid", label = "Mean accuracy", linewidth = 1)
    
    for i, ax in enumerate(axes):
        ax.set_ylim(0, 100)
        ax.set_xlabel("Participants")
        ax.set_ylabel("Percent correct")
        ax.set_title(f"{group_labels[i]}")
        # remove the xticks
        ax.set_xticks([])

    
    # plot the chance level
    if chance_level:
        for ax in axes:
            ax.axhline(chance_level, color = "black", linestyle = "dashed", label = "Chance level", linewidth = 1)

    for ax in axes:
        ax.legend()


   #for ax in axes:
    #    ax.axhline(np.mean(percent_correct), color = "black", linestyle = "solid", label = "Mean accuracy", linewidth = 0.5)
    
    # group means
    #if groups:
    #    for group in group_labels:
    #        group_inds = [ind for ind, g in enumerate(groups) if g == group]
    #        group_mean = np.mean([percent_correct[ind] for ind in group_inds])
    #        ax.axhline(group_mean, color = colours[group], linestyle = "solid", label = group_labels[group], linewidth = 0.5)

    
    # add labels for legend
    #if group_labels:
    #    for group in group_labels:
    #        ax.bar([0], [0], color = colours[group], label = group_labels[group])
    
    #if chance_level or group_labels:
    #    ax.legend()

    
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath)

    plt.close()



def plot_recoveries(trues:list, estimateds:list, parameter_names:list, savepath: Path, standardize:bool = False):
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
    standardize : bool, optional
        If True, standardize the estimated parameters. The default is False.
    
    Returns
    -------
    None
    """
    fig, axes = plt.subplots(2, len(trues) // 2 + (len(trues) % 2 > 0), figsize = (10, 7), dpi = 300)
    
    for true, estimated, parameter_name, axis in zip(trues, estimateds, parameter_names, axes.flatten()):
        if standardize:
            # normalise in the range of the true parameter
            estimated = (estimated - np.min(estimated)) / (np.max(estimated) - np.min(estimated)) * (np.max(true) - np.min(true)) + np.min(true)
            y_label = "Estimated (scaled)"
        else:
            y_label = "Estimated"
        plot_recovery_ax(axis, true, estimated, parameter_name, y_label)

    # if any of the axes is empty, remove it
    for axis in axes.flatten():
        if not axis.get_title():
            fig.delaxes(axis)
    
    plt.tight_layout()
    
    if savepath:
        plt.savefig(savepath)

    plt.close()

def plot_recovery_ax(ax, true, estimated, parameter_name, ylabel = "Estimated"):
    """
    Helper function for plot_recoveries
    """
    ax.scatter(true, estimated, s=10, color = "steelblue")
    x_lims = ax.get_xlim()
    y_lims = ax.get_ylim()
    ax.plot([y_lims[0], x_lims[1]], [y_lims[0], x_lims[1]], color = "black", linestyle = "dashed")

    # plot the correlation between the true and estimated parameters
    # get intercept and slope of the regression line
    corr = np.corrcoef(true, estimated)[0, 1]
    intercept = np.mean(estimated) - corr*np.mean(true)


    # plot a correlation line
    lin_space = np.linspace(x_lims[0], x_lims[1], 100)
    ax.plot(lin_space, intercept + corr*lin_space, color = "steelblue", linestyle = "solid", linewidth = 0.5)

    ax.set_xlabel("True")
    ax.set_ylabel(ylabel)
    ax.set_title(parameter_name)



def plot_posteriors_violin(posteriors, parameter_names, trues = None, savepath = None):
    """
    Plot the posterior densities of a set of parameters as violin plots.
    
    Parameters
    ----------
    posteriors : list of np.arrays
        List of posteriors.
    parameter_names : list of str
        List of parameter names.
    trues : list of float, optional
        List of true parameters. The default is None.
    savepath : Path, optional
        Path to save the figure to. The default is None.
    """
    fig, ax = plt.subplots(1, 1, figsize = (7, 5), dpi = 300)

    # line at 0
    ax.axhline(0, color = "black", linestyle = "dashed", linewidth = 0.5)

    ax.violinplot(posteriors, showextrema = False, widths=0.8)

    # plot quantiles as whiskers
    for i, density in enumerate(posteriors):
        quantiles = np.quantile(density, [0.025, 0.975])
        ax.plot([i+1, i+1], quantiles, color = "midnightblue", linewidth = 2)

    # plot median with a white dot
    medians = [np.median(density) for density in posteriors]
    ax.scatter(range(1, len(medians)+1), medians, color = "white", s = 5, zorder = 3)

    # plot the true parameter values
    if trues:
        ax.scatter(range(1, len(trues)+1), trues, color = "red", s = 5, zorder = 3)

    # plot the maximum posterior density
    mpds = [maximum_posterior_density(density) for density in posteriors]
    ax.scatter(range(1, len(mpds)+1), mpds, color = "pink", s = 5, zorder = 3)

    ax.set_xticks(range(1, len(parameter_names)+1))
    ax.set_xticklabels(parameter_names)

    # aesthetics to make the plot look nicer
    ax.set_ylabel("Posterior density")
    ax.set_xlabel("Parameter")

    plt.tight_layout()    

    if savepath:
        plt.savefig(savepath)

    plt.close()




def plot_compare_posteriors(posteriors_1:list[list[float]], posteriors_2:list[list[float]], parameter_names:list[str], group_labels:list[str], savepath: Path = None):
    """
    Plot the posterior densities of two sets of parameters to compare them.

    Parameters
    ----------
    posteriors_1 : list of np.arrays
        List of posteriors.
    posteriors_2 : list of np.arrays
        List of posteriors.
    parameter_names : list of str   
        List of parameter names.
    group_labels : list of str
        List of group labels.
    savepath : Path, optional
        Path to save the figure to. The default is None.
    """

    fig, axes = plt.subplots(2, 3, figsize = (10, 7), dpi = 300)
    
    for posterior_1, posterior_2, parameter_name, ax in zip(posteriors_1, posteriors_2, parameter_names, axes.flatten()):
        plot_compare_posterior_ax(ax, posterior_1, posterior_2, parameter_name, group_labels)

    
    # legend on the last axis
    axes[-1, -1].legend()
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath)

    plt.close()

def plot_compare_posterior_ax(ax, posterior_1, posterior_2, parameter_name, group_labels):
    """
    Helper function for plot_compare_posteriors
    """
    
    # use seaborn for density plots
    sns.kdeplot(posterior_1, ax = ax, color = colours[0], fill = True, label = group_labels[0])
    sns.kdeplot(posterior_2, ax = ax, color = colours[1], fill = True, label = group_labels[1])

    ax.set_title(parameter_name)



def plot_compare_prior_posteriors(priors:list[list[float]], posteriors_1:list[list[float]], posteriors_2:list[list[float]], parameter_names:list[str], group_labels:list[str], savepath: Path = None):
    """
    Plot the posterior densities of two sets of parameters to compare them.

    Parameters
    ----------
    priors : list of np.arrays
        List of priors.
    posteriors_1 : list of np.arrays
        List of posteriors.
    posteriors_2 : list of np.arrays
        List of posteriors.
    parameter_names : list of str   
        List of parameter names.
    group_labels : list of str
        List of group labels.
    savepath : Path, optional
        Path to save the figure to. The default is None.
    """

    fig, axes = plt.subplots(2, 3, figsize = (10, 7), dpi = 300)
    
    for prior, posterior_1, posterior_2, parameter_name, ax in zip(priors, posteriors_1, posteriors_2, parameter_names, axes.flatten()):
        plot_compare_prior_posterior_ax(ax, prior, posterior_1, posterior_2, parameter_name, group_labels)
    
    # legend on the last axis
    axes[-1, -1].legend()
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath)

    plt.close()

def plot_compare_prior_posterior_ax(ax, prior, posterior_1, posterior_2, parameter_name, group_labels):
    """
    Helper function for plot_compare_posteriors
    """
    
    # use seaborn for density plots
    sns.kdeplot(prior, ax = ax, color = "black", linestyle = "dashed", fill = False, label = "Prior", linewidth = 1)
    sns.kdeplot(posterior_1, ax = ax, color = colours[0], fill = True, label = group_labels[0])
    sns.kdeplot(posterior_2, ax = ax, color = colours[1], fill = True, label = group_labels[1])

    ax.set_title(parameter_name)



def plot_posterior_ax(ax, prior, posterior, parameter_name):
    sns.kdeplot(prior, ax = ax, color = "black", linestyle = "dashed", fill = False, label = "Prior", linewidth = 1)
    
    credible_interval = np.quantile(posterior, [0.025, 0.975])
    
    # only plot the credible interval
    sns.kdeplot(posterior, ax = ax, color = "steelblue", fill = True, label = "Posterior", clip = (credible_interval[0], credible_interval[1]), alpha = 0.4, linewidth = 0.00001)

    # plot posterior with different colors for the credible interval
    sns.kdeplot(posterior, ax = ax, color = "steelblue", fill = False, label = "Posterior", alpha = 1, linewidth = 2)
    

    ax.set_title(parameter_name)

def plot_posterior(priors:list[list[float]], posteriors:list[list[float]], parameter_names:[list[str]], savepath: Path = None):

    fig, axes = plt.subplots(3, 2, figsize = (9, 10), dpi = 300)

    for prior, posterior, ax, param in zip(priors, posteriors, axes.flatten(), parameter_names):
        plot_posterior_ax(ax, prior, posterior, param)


    # dashed line for the prior, solid line for the posterior, and fill the area in between 
    custom_lines = [plt.Line2D([0], [0], color = "black", linestyle = "dashed", lw = 1),
                    plt.Line2D([0], [0], color = "steelblue", lw = 2),
                    Patch(facecolor = "steelblue", alpha = 0.4)]

    axes[0, 0].legend(custom_lines, ["Prior", "Posterior", "95% CI"])

    # if there is an empty axis, remove it
    for ax in axes.flatten():
        if not ax.get_title():
            fig.delaxes(ax)

    # make legend on the last axis
    
    # prep stuff to put in the legend
    


    plt.tight_layout()

    if savepath:
        plt.savefig(savepath)



