import numpy as np
import pandas as pd


def softmax(x, theta):
    """
    Softmax function.

    Parameters
    ----------
    x : numpy.ndarray
        Array of values.
    theta : float
        Inverse temperature parameter.
    
    Returns
    -------
    softmax : numpy.ndarray
        Softmaxed values.
    """
    exp_x = np.exp(x*theta)
    return exp_x / np.sum(exp_x)


def create_payoff_structure(
        n_trials : int = 100, 
        freq : float = 0.5,
        infreq : float = 0.1,
        bad_r : int = 100,
        bad_freq_l : int = -250,
        bad_infreq_l : int = -1250,
        good_r : int = 50,
        good_freq_l : int = -50,
        good_infreq_l : int = -250
        ):
    """
    Creates a payoff structure for the ORL task.

    Parameters
    ----------
    n_trials : int
        Total number of trials in our payoff structure. Must be divisible by 10.
    freq : float
        Probability of our frequent losses (we have losses half of the time).
    infreq : float
        Probability of our infrequent losses (we have losses 1/10th of the time).
    bad_r : int
        "Bad" winnings.
    bad_freq_l : int
        "Bad" frequent loss.
    bad_infreq_l : int
        "Bad" infrequent loss.
    good_r : int
        "Good" winnings.
    good_freq_l : int
        "Good" frequent loss.
    good_infreq_l : int
        "Good" infrequent loss.

    Returns
    -------
    payoff : numpy.ndarray
        A payoff structure for the ORL task.
    """
    # check that number of trials is divisible by 10
    if n_trials % 10 != 0:
        raise ValueError("n_trials must be divisible by 10")
    
    n_struct = int(n_trials/10) # size of our subdivisions for pseudorandomization

    def create_deck(R, L, prob, n=10):
        R_deck = np.repeat(R, n) # we win on every trial
        L_deck = np.concatenate(
            (np.repeat(L, int(n * prob)), np.repeat(0, int(n * (1 - prob)))) # we have losses with probability prob
        )
        return R_deck + np.random.choice(L_deck, size=n, replace=False)

    
    A = np.hstack([create_deck(bad_r, bad_freq_l, freq) for i in range(n_struct)]) # bad frequent

    B = np.hstack([create_deck(bad_r, bad_infreq_l, infreq) for i in range(n_struct)]) # bad infrequent
    
    C = np.hstack([create_deck(good_r, good_freq_l, freq) for i in range(n_struct)]) # good frequent

    D = np.hstack([create_deck(good_r, good_infreq_l, infreq) for i in range(n_struct)]) # good infrequent

    payoff = np.column_stack((A,B,C,D))/100

    return payoff


def simulate_ORL(
        a_rew : float, 
        a_pun : float, 
        K : float,
        omega_f : float,
        omega_p : float, 
        theta : float,
        payoff : np.ndarray = create_payoff_structure(), 
        n_trials : int = 100, 

        ):
    """
    Simulates behavioural data using the payoff structure and the ORL model.

    Parameters
    ----------
    payoff : numpy.ndarray
        A payoff structure for the ORL task.
    n_trials : int
        Total number of trials in our payoff structure. Must be divisible by 10.
    a_rew : float
        Learning rate for rewards.
    a_pun : float
        Learning rate for punishments.
    K : float
        Perseveration parameter.
    omega_f : float
        Weighting parameter for expected frequencies.
    omega_p : float
        Weighting parameter for perseveration.
    theta : float
        Inverse temperature parameter.
    
    Returns
    -------
    data : dict
        A dictionary containing the simulated data.
    """
    choices = np.zeros(n_trials).astype(int)
    outcomes = np.zeros(n_trials)
    sign_out = np.zeros(n_trials)

    # setting initial values
    ev = np.zeros(4)
    perseverance = np.ones(4) / (1 + K)
    ef = np.zeros(4)
    util = softmax(np.ones(4)*0.25, theta)

    # looping over trials
    for t in range(n_trials):
        # choice
        choices[t] = np.random.choice(4, p=util)

        # outcome
        outcomes[t] = payoff[t, int(choices[t])]

        # get the sign of the reward
        sign_out[t] = np.sign(outcomes[t])


        ## prediction error
        PEval = outcomes[t] - ev[choices[t]]
        PEfreq = sign_out[t] - ef[choices[t]]
        PEfreq_fic = -sign_out[t]/3 - ef

        # store chosen deck expected value
        ef_chosen = ef[choices[t]]
        ev_chosen = ev[choices[t]]

        if outcomes[t] >= 0: # if the outcome is zero or above
            # update ev for all decks
            ef += a_pun * PEfreq_fic

            # update ev for chosen deck using the stored value
            ef[choices[t]] = ef_chosen + a_rew * PEfreq
            ev[choices[t]] = ev_chosen + a_rew * PEval
        else: # if the outcome is negative
            # update ev for all decks
            ef += a_rew * PEfreq_fic

            # update ev for chosen deck using the stored value
            ef[choices[t]] = ef_chosen + a_pun * PEfreq
            ev[choices[t]] = ev_chosen + a_pun * PEval
        
        # perseverance update  
        perseverance[choices[t]] = 1 # set the chosen deck to 1
        perseverance /= (1 + K)    

        # update utility
        util = softmax(ev + omega_f * ef + omega_p * perseverance, theta)

    data = {
        "choice" : choices.astype(int) + 1,
        "outcome" : outcomes,
        "trial":  range(1, n_trials + 1),
        "sign_out": sign_out
    }

    return data


def simulate_ORL_group(
        n_subjects : int = 10,
        n_trials : int = 100,
        mu_a_rew : float = 0.3,
        sigma_a_rew : float = 0.05,
        mu_a_pun : float = 0.3,
        sigma_a_pun : float = 0.05,
        mu_K : float = 0.3,
        sigma_K : float = 0.05,
        mu_omega_f : float = 0.7,
        sigma_omega_f : float = 0.05,
        mu_omega_p : float = 0.7,
        sigma_omega_p : float = 0.05,
        mu_theta : float = 1.,
        sigma_theta : float = 0.05
        ):
    """
    Simulates behavioural data using the payoff structure and the ORL model given a group level mean
    for the parameters.

    Parameters
    ----------
    n_subjects : int
        Number of subjects in the group
    n_trials : int
        Total number of trials in our payoff structure. Must be divisible by 10.
    
    Returns
    -------
    data : dict
        A dictionary containing the simulated data.
    """

    choices, outcomes, sign_out, trial = np.zeros((n_subjects, n_trials)), np.zeros((n_subjects, n_trials)), np.zeros((n_subjects, n_trials)), np.zeros((n_subjects, n_trials))
    sub_a_rew, sub_a_pun, sub_K = np.zeros(n_subjects), np.zeros(n_subjects), np.zeros(n_subjects)
    sub_omega_f, sub_omega_p, sub_theta = np.zeros(n_subjects), np.zeros(n_subjects), np.zeros(n_subjects)


    for sub in range(n_subjects):
        # generate parameters
        sub_a_rew[sub] = np.random.normal(mu_a_rew, sigma_a_rew)
        sub_a_pun[sub] = np.random.normal(mu_a_pun, sigma_a_pun)

        # check that parameters are between 0 and 1
        while sub_a_rew[sub] < 0 or sub_a_rew[sub] > 1:
            sub_a_rew[sub] = np.random.normal(mu_a_rew, sigma_a_rew)
        while sub_a_pun[sub] < 0 or sub_a_pun[sub]  > 1:
            sub_a_pun[sub] = np.random.normal(mu_a_pun, sigma_a_pun)
        

        sub_K[sub] = np.random.normal(mu_K, sigma_K)
        sub_omega_f[sub] = np.random.normal(mu_omega_f, sigma_omega_f)
        sub_omega_p[sub] = np.random.normal(mu_omega_p, sigma_omega_p)
        sub_theta[sub] = np.random.normal(mu_theta, sigma_theta)

        # check that the parameters K and theta are between 0 and 5
        while sub_K[sub] < 0 or sub_K[sub] > 5:
            sub_K[sub] = np.random.normal(mu_K, sigma_K)
        while sub_theta[sub] < 0 or sub_theta[sub] > 5:
            sub_theta[sub] = np.random.normal(mu_theta, sigma_theta)

        # simulate data
        payoff = create_payoff_structure(n_trials=n_trials)

        sub_data = simulate_ORL(
            payoff = payoff, 
            n_trials = n_trials, 
            a_rew = sub_a_rew[sub], 
            a_pun = sub_a_pun[sub], 
            K = sub_K[sub], 
            omega_f = sub_omega_f[sub], 
            omega_p = sub_omega_p[sub],
            theta = sub_theta[sub]
            )

        choices[sub] = sub_data["choice"]
        outcomes[sub] = sub_data["outcome"]
        sign_out[sub] = sub_data["sign_out"]
        trial[sub] = sub_data["trial"]

    data = {
        "choice": choices.astype(int).flatten(),
        "outcome": outcomes.flatten(),
        "sign_out": sign_out.flatten(),
        "trial": trial.flatten(),
        "sub": np.repeat(np.arange(1, n_subjects + 1), n_trials),
        "sub_a_rew": np.repeat(sub_a_rew, n_trials),
        "sub_a_pun": np.repeat(sub_a_pun, n_trials),
        "sub_K": np.repeat(sub_K, n_trials),
        "sub_omega_f": np.repeat(sub_omega_f, n_trials),
        "sub_omega_p": np.repeat(sub_omega_p, n_trials)
    }

    # flatten the two dimensional arrays
    for var in ["choice", "outcome", "sign_out", "trial"]:
        data[var] = data[var].flatten()

    # make into a dataframe
    df = pd.DataFrame.from_dict(data)

    return df