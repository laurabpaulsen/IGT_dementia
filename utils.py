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

    payoff = np.column_stack((A,B,C,D))/100 # combining all four decks as columns with each 100 trials - dividing our payoffs by 100 to make the numbers a bit easier to work with

    return payoff


def simulate_ORL(
        payoff : np.ndarray = create_payoff_structure(), 
        n_trials : int = 100, 
        a_rew : float = 0.3, 
        a_pun : float = 0.3, 
        K : float = 3,
        omega_f : float = 0.7, 
        omega_p : float = 0.7
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
    perseverance = np.zeros(4)
    ef = np.zeros(4)

    # looping over trials
    for t in range(n_trials):
        valence = ev + omega_f * ef + omega_p * perseverance
        
        # probability of choosing each deck (softmax)
        exp_p = np.exp(valence)
        p = exp_p / np.sum(exp_p)

        # choice
        choices[t] = np.random.choice(4, p=p)
        
        # outcome
        outcomes[t] = payoff[t, int(choices[t])]

        # get the sign of the reward
        sign_out[t] = np.sign(outcomes[t])

        # update perseveration
        # set perseveration to 1 if the deck was chosen
        perseverance[choices[t]] = 1
        perseverance = perseverance / (1 + K)
        
        # update expected value for chosen deck
        if sign_out[t] == 1:
            ev[choices[t]] = ev[choices[t]] + a_rew * (outcomes[t] - ev[choices[t]])
        else:
            ev[choices[t]] = ev[choices[t]] + a_pun * (outcomes[t] - ev[choices[t]])
        
        # update expected frequency for chosen deck
        if sign_out[t] == 1:
            ef[choices[t]] = ef[choices[t]] + a_rew * (1 - ef[choices[t]])
        else:
            ef[choices[t]] = ef[choices[t]] + a_pun * (1 - ef[choices[t]])
        
        # update expected frequency for unchosen decks (fictive frequencies)
        for d in range(4):
            if d != int(choices[t]):
                if sign_out[t] == 1:
                    ef[d] = ef[d] + a_rew * (sign_out[t]/3 * -ef[d])
                else:
                    ef[d] = ef[d] + a_pun * (sign_out[t]/3 * -ef[d])

    data = {
        "choice" : choices.astype(int) + 1,
        "outcome" : outcomes,
        "trial":  range(1, n_trials + 1),
        "sign_out": sign_out
    }

    return data


