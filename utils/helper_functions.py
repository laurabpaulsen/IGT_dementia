from scipy.stats import binom
import numpy as np
import argparse

def parse_n_subj():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_subj",
        "-n",
        type = int,
        default = 10,
        help = "The number of subjects to simulate, recover or plot depending on the script."
    )
    args = parser.parse_args()
    return args.n_subj


def parse_n_subj_groups():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_subj",
        "-n",
        type = int,
        default = 10,
        help = "The number of subjects to simulate, recover or plot depending on the script."
    )
    parser.add_argument(
        "--n_groups",
        "-g",
        type = int,
        default = 2,
        help = "The number of groups to simulate, recover or plot depending on the script."
    )
    args = parser.parse_args()
    return args.n_subj, args.n_trials

def logit(x):
    return np.log(x / (1 - x))

def inv_logit(x):
    return 1 / (1 + np.exp(-x))


def chance_level(n, alpha = 0.001, p = 0.5):
    """
    Calculates the chance level for a given number of trials and alpha level

    Parameters
    ----------
    n : int
        The number of trials.
    alpha : float
        The alpha level.
    p : float
        The probability of a correct response.

    Returns
    -------
    chance_level : float
        The chance level.
    """
    k = binom.ppf(1-alpha, n, p)
    chance_level = k/n
    
    return chance_level