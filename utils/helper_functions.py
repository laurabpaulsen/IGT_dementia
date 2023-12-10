from scipy.stats import binom
import numpy as np
import argparse
from scipy.stats import norm

def maximum_posterior_density(posterior: np.array) -> float:
    """
    Calculates the maximum posterior density for a given posterior distribution.

    Parameters
    ----------
    posterior : np.array
        The posterior samples

    Returns
    -------
    maximum_posterior_density : float
        The maximum posterior density.
    """
    
    """
    Andreas' R code:
    MPD <- function(x) {
        density(x)$x[which(density(x)$y==max(density(x)$y))]
    }"""

    # get the density of the posterior
    density = np.histogram(posterior, bins = 100, density = True)
    
    # get the index of the maximum density
    max_index = np.argmax(density[0])

    # get the maximum posterior density
    maximum_posterior_density = density[1][max_index]
    
    return maximum_posterior_density

def probit(p):
    return norm.ppf(p)

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
    return args.n_subj, args.n_groups

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