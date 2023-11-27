"""
This scipt holds the functions for generating the data for the parameter recovery analysis.

If the script is run directly, it will generate the data and save it to a csv file.
"""
import numpy as np
import pandas as pd
from pathlib import Path

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

def simulate_ORL_group(
        n_subjects : int = 10,
        n_trials : int = 100,
        mu_a_rew : float = 0.3,
        sigma_a_rew : float = 0.1,
        mu_a_pun : float = 0.3,
        sigma_a_pun : float = 0.1,
        mu_K : float = 0.3,
        sigma_K : float = 0.1,
        mu_omega_f : float = 0.7,
        sigma_omega_f : float = 0.1,
        mu_omega_p : float = 0.7,
        sigma_omega_p : float = 0.1
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

    choices = np.zeros((n_subjects, n_trials))
    outcomes = np.zeros((n_subjects, n_trials))
    sign_out = np.zeros((n_subjects, n_trials))
    sub_a_rew = np.zeros(n_subjects)
    sub_a_pun = np.zeros(n_subjects)
    sub_K = np.zeros(n_subjects)
    sub_omega_f = np.zeros(n_subjects)
    sub_omega_p = np.zeros(n_subjects)



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

        # check that the parameters are < 0
        while sub_K[sub] < 0:
            sub_K[sub] = np.random.normal(mu_K, sigma_K)
        while sub_omega_f[sub] < 0:
            sub_omega_f[sub] = np.random.normal(mu_omega_f, sigma_omega_f)
        while sub_omega_p[sub] < 0:
            sub_omega_p[sub] = np.random.normal(mu_omega_p, sigma_omega_p)

        # simulate data
        payoff = create_payoff_structure(n_trials=n_trials)

        sub_data = simulate_ORL(payoff, n_trials, sub_a_rew[sub], sub_a_pun[sub], sub_K[sub], sub_omega_f[sub], sub_omega_p[sub])

        choices[sub] = sub_data["choice"]
        outcomes[sub] = sub_data["outcome"]
        sign_out[sub] = sub_data["sign_out"]

    data = {
        "choice" : choices.astype(int),
        "outcome" : outcomes,
        "sign_out" : sign_out,
        "sub" : [sub + 1 for sub in range(n_subjects) for i in range(n_trials)],
        "sub_a_rew" : [sub_a_rew[sub] for sub in range(n_subjects) for i in range(n_trials)],
        "sub_a_pun" : [sub_a_pun[sub] for sub in range(n_subjects) for i in range(n_trials)],
        "sub_K" : [sub_K[sub] for sub in range(n_subjects) for i in range(n_trials)],
        "sub_omega_f" : [sub_omega_f[sub] for sub in range(n_subjects) for i in range(n_trials)],
        "sub_omega_p" : [sub_omega_p[sub] for sub in range(n_subjects) for i in range(n_trials)]
    }

    # prep for dataframe conversion
    data["choice"] = data["choice"].flatten()
    data["outcome"] = data["outcome"].flatten()
    data["sign_out"] = data["sign_out"].flatten()



    # make into a dataframe
    df = pd.DataFrame.from_dict(data)

    return df


def simulate_ORL(
        payoff : np.ndarray = create_payoff_structure(), 
        n_trials : int = 100, 
        a_rew : float = 0.3, 
        a_pun : float = 0.3, 
        K : float = 3,
        theta : float = 3, 
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
    theta : float
        Inverse temperature parameter.
    omega_f : float
        Weighting parameter for expected frequencies.
    omega_p : float
        Weighting parameter for perseveration.
    
    Returns
    -------
    data : dict
        A dictionary containing the simulated data.
    

    """

    choices = np.zeros(n_trials)
    outcomes = np.zeros(n_trials)
    sign_out = np.zeros(n_trials)

    ev = np.zeros((n_trials, 4))
    perseverance = np.zeros((n_trials, 4))
    exp_freq = np.zeros((n_trials, 4))
    valence = np.zeros((n_trials, 4))

    exp_p = np.zeros((n_trials, 4))
    p = np.zeros((n_trials, 4))

    # initial values
    ev[0] = np.zeros(4)
    exp_freq[0] = np.zeros(4)
    perseverance[0] = np.ones(4) / (1 - K)

    # initial choice
    choices[0] = np.random.choice(4, p=np.ones(4) / 4)
    outcomes[0] = payoff[0, int(choices[0])]


    # looping over trials
    for t in range(1, n_trials):
        # get the sign of the reward on the previous trial
        sign_out[t] = np.sign(outcomes[t - 1])

        for d in range(4):
            if d != int(choices[t - 1]): # if the deck was not chosen
                ev[t, d] = ev[t - 1, d] # expected value stays the same
                perseverance[t, d] = perseverance[t - 1, d]/(1 + K) # perseverance decays

                if sign_out[t] == 1:
                    exp_freq[t, d] = exp_freq[t - 1, d] + a_rew * (-exp_freq[t - 1, d])
                else:
                    exp_freq[t, d] = exp_freq[t - 1, d] + a_pun * (-exp_freq[t - 1, d])


            else: # if the deck was chosen
                perseverance[t, d] = 1 / (1 + K) # perseverance resets

                if sign_out[t] == 1: # if the reward was positive
                    ev[t, d] = ev[t - 1, d] + a_rew * (outcomes[t - 1] - ev[t - 1, d])
                    exp_freq[t, d] = exp_freq[t - 1, d] + a_rew * (1 - exp_freq[t - 1, d])
                else: # if the reward was negative
                    ev[t, d] = ev[t - 1, d] + a_pun * (outcomes[t - 1] - ev[t - 1, d])
                    exp_freq[t, d] = exp_freq[t - 1, d] + a_pun * (1 - exp_freq[t - 1, d])

            
            # valence model
            valence[t, d] = ev[t, d] + omega_f * exp_freq[t, d] + omega_p * perseverance[t, d]

        # softmax
        exp_p[t] = np.exp(theta * valence[t])
        p[t] = exp_p[t] / np.sum(exp_p[t])

        # check if any numbers are NaN, if they are, set them to 0 but only for the nan values
        if np.isnan(p[t]).any():
            p[t][np.isnan(p[t])] = 0
            print("NaNs set to 0")
            
        # choice
        choices[t] = np.random.choice(4, p=p[t])
        outcomes[t] = payoff[t, int(choices[t])]
    
    data = {
        "choice" : [int(choice) + 1 for choice in choices],
        "outcome" : [int(outcome) + 1 for outcome in outcomes],
        "T": int(n_trials),
        "sign_out": sign_out
    }

    return data
        

if __name__ in "__main__":
    path = Path(__file__).parent

    # output path for simulated data
    output_path = path / "simulated"

    # create output path if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame()
    
    n_groups = 10 
    n_subjects = 20

    for group in range(n_groups):
        mu_a_rew = np.random.uniform(0, 1)
        mu_a_pun = np.random.uniform(0, 1)


        mu_K = np.random.uniform(0, 10)

        mu_omega_f = np.random.uniform(0, 5)
        mu_omega_p = np.random.uniform(0, 5)


        data = simulate_ORL_group(
            n_subjects = n_subjects,
            mu_a_rew = mu_a_rew,
            mu_a_pun = mu_a_pun,
            mu_K = mu_K,
            mu_omega_f = mu_omega_f,
            mu_omega_p = mu_omega_p,
            sigma_a_rew = 0.001,
            sigma_a_pun = 0.001,
            sigma_K = 0.001,
            sigma_omega_f = 0.001,
            sigma_omega_p = 0.001
            )
    
        tmp_df = pd.DataFrame.from_dict(data)
        tmp_df["group"] = group + 1
        tmp_df["mu_a_rew"] = mu_a_rew
        tmp_df["mu_a_pun"] = mu_a_pun
        tmp_df["mu_K"] = mu_K
        tmp_df["mu_omega_f"] = mu_omega_f
        tmp_df["mu_omega_p"] = mu_omega_p

        df = pd.concat([df, tmp_df])

    df.to_csv(output_path / f"ORL_simulated_{n_groups}_groups_{n_subjects}_sub.csv", index=False)


    






