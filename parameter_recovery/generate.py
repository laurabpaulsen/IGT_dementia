"""
This scipt holds the functions for generating the data for the parameter recovery analysis.
"""
import numpy as np

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
    payoff = create_payoff_structure()
    data = simulate_ORL(payoff)
    print(data)
    



"""
ORL <- function(payoff,ntrials,a_rew,a_pun,K,theta,omega_f,omega_p) {
  
  for (t in 2:ntrials) {
    
    #this is important mention this as constructing model
    signX[t] <- ifelse(X[t-1]<0,-1,1)
    
    for (d in 1:4) {
      
      #-----------Valence model------------------------------
      valence[t,d] <- ev[t,d] + Ef[t,d]*omega_f + PS[t,d]*omega_p
      
      #----------softmax part 1-------------
      exp_p[t,d] <- exp(theta*valence[t,d])
      
    }
    
    #----------softmax part 2-------------
    for (d in 1:4) {
      p[t,d] <- exp_p[t,d]/sum(exp_p[t,])
    }
      
    x[t] <- rcat(1,p[t,])
    
    X[t] <- payoff[t,x[t]]
    
  }
  
  result <- list(x=x,
                 X=X,
                 Ev=Ev,
                 Ef=Ef,
                 PS=PS)
  
  return(result)
  
  
}

"""










"""

#-------test ORL delta function and jags script ---------

#---set params

a_rew <- .3
a_pun <- .3
K <- 2
theta <- 2
omega_f <- .7
omega_p <- .7

# ntrials <- 100

source("ORL.R")
ORL_sims <- ORL(payoff,ntrials,a_rew,a_pun,K,theta,omega_f,omega_p)

par(mfrow=c(2,2))
plot(ORL_sims$ev[,1])
plot(ORL_sims$ev[,2])
plot(ORL_sims$ev[,3])
plot(ORL_sims$ev[,4])

x <- ORL_sims$x
X <- ORL_sims$X

# set up jags and run jags model
data <- list("x","X","ntrials") 
params<-c("a_rew","a_pun","K","theta","omega_f","omega_p")
samples <- jags.parallel(data, inits=NULL, params,
                model.file ="ORL.txt", n.chains=3, 
                n.iter=5000, n.burnin=1000, n.thin=1, n.cluster=3)


###--------------Run full parameter recovery -------------
niterations <- 100 # fewer because it takes too long

true_a_rew <- array(NA,c(niterations))
true_a_pun <- array(NA,c(niterations))
true_K <- array(NA,c(niterations))
true_theta <- array(NA,c(niterations))
true_omega_f <- array(NA,c(niterations))
true_omega_p <- array(NA,c(niterations))

infer_a_rew <- array(NA,c(niterations))
infer_a_pun <- array(NA,c(niterations))
infer_K <- array(NA,c(niterations))
infer_theta <- array(NA,c(niterations))
infer_omega_f <- array(NA,c(niterations))
infer_omega_p <- array(NA,c(niterations))

start_time = Sys.time()

for (i in 1:niterations) {
  
  # let's see how robust the model is. Does it recover all sorts of values?
  a_rew <- runif(1,0,1)
  a_pun <- runif(1,0,1)
  K <- runif(1,0,2)
  theta <- runif(1,.2,3) # could also just be a set value (e.g. 1) to simplify the model a bit
  omega_f <- runif(1,-2,2)
  omega_p <- runif(1,-2,2)
  
  ORL_sims <- ORL(payoff,ntrials,a_rew,a_pun,K,theta,omega_f,omega_p)
  
  x <- ORL_sims$x
  X <- ORL_sims$X
  
  # set up jags and run jags model
  data <- list("x","X","ntrials") 
  params<-c("a_rew","a_pun","K","theta","omega_f","omega_p")
  samples <- jags.parallel(data, inits=NULL, params,
                  model.file ="ORL.txt", n.chains=3, 
                  n.iter=3000, n.burnin=1000, n.thin=1, n.cluster=3)
  
  
  true_a_rew[i] <- a_rew
  true_a_pun[i] <- a_pun
  true_K[i] <- K
  true_theta[i] <- theta
  true_omega_f[i] <- omega_f
  true_omega_p[i] <- omega_p
  
  # find maximum a posteriori
  Y <- samples$BUGSoutput$sims.list
  infer_a_rew[i] <- MPD(Y$a_rew)
  infer_a_pun[i] <- MPD(Y$a_pun)
  infer_K[i] <- MPD(Y$K)
  infer_theta[i] <- MPD(Y$theta)
  infer_omega_f[i] <- MPD(Y$omega_f)
  infer_omega_p[i] <- MPD(Y$omega_p)

  print(i)
  
}

end_time = Sys.time()
end_time - start_time

# let's look at some scatter plots

par(mfrow=c(3,2))
plot(true_a_rew,infer_a_rew)
plot(true_a_pun,infer_a_pun)
plot(true_K,infer_K)
plot(true_theta,infer_theta)
plot(true_omega_f,infer_omega_f)
plot(true_omega_p,infer_omega_p)

# plotting code courtesy of Lasse
source('recov_plot.R')
pl1 <- recov_plot(true_a_rew, infer_a_rew, c("true a_rew", "infer a_rew"), 'smoothed linear fit')
pl2 <- recov_plot(true_a_pun, infer_a_pun, c("true a_pun", "infer a_pun"), 'smoothed linear fit')
pl3 <- recov_plot(true_K, infer_K, c("true K", "infer K"), 'smoothed linear fit')
pl4 <- recov_plot(true_theta, infer_theta, c("true theta", "infer theta"), 'smoothed linear fit')
pl5 <- recov_plot(true_omega_f, infer_omega_f, c("true omega_f", "infer omega_f"), 'smoothed linear fit')
pl6 <- recov_plot(true_omega_p, infer_omega_p, c("true omega_p", "infer omega_p"), 'smoothed linear fit')
ggarrange(pl1, pl2, pl3, pl4, pl5, pl6)

# for investigating multi-colinearity

# par(mfrow=c(2,2))
# plot(true_a_rew,true_a_pun)
# plot(infer_a_rew,infer_a_pun)
# plot(true_omega_f,true_omega_p)
# plot(infer_omega_f,infer_omega_p)
# 
# par(mfrow=c(2,2))
# plot(true_a_rew,true_omega_f)
# plot(infer_a_rew,infer_omega_f)
# plot(true_a_rew,true_omega_p)
# plot(infer_a_rew,infer_omega_p)

"""