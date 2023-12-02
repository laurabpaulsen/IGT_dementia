from scipy.stats import binom
import numpy as np

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