
data {
  int<lower=1> N; // number of subjecs
  int<lower=1> T; // maxiumum number of trials
  array[N] int<lower=1, upper=T> Tsubj; // trials per subject
  array[N, T] int choice; // choices made
  array[N, T] real outcome; // oucomes
  array[N, T] real sign_out; // sign of outcome
  array[N] int group; // which group the subject belongs to (1 or 2)
}

transformed data {
  vector[4] initV;
  initV  = rep_vector(0.0, 4);
}

parameters {
// Declare all parameters as vectors for vectorizing
  // Hyper(group)-parameters
  vector[6] mu; // overall mean (in both groups)
  vector[6] delta; // the difference btween the groups
  vector<lower=0>[6] sigma_group1;
  vector<lower=0>[6] sigma_group2;


  // Subject-level parameters 
  vector<lower=0, upper=1>[N]   a_rew;
  vector<lower=0, upper=1>[N]   a_pun;
  vector<lower=0>[N]            K;
  vector<lower=0>[N]            theta;
  vector[N]                     omega_f;
  vector[N]                     omega_p;
}

model {
  // Hyperparameters
  //CHANGED (truncated mu)
  mu ~ normal(0, 1);
  delta[1:2] ~ normal(0, 1) T[-1, 1];
  delta[3:6] ~ normal(0, 1);

  // group level parameters
  sigma_group1 ~ gamma(.1,.1);
  sigma_group2 ~ gamma(.1,.1);

  // modelling each parameter for each subject according to the overall mean and the difference between the groups (with a group level standard deviation)
  for(n in 1:N){
    if (group[n] == 1){ 
      a_rew[n]    ~ normal((mu[1] - (delta[1]/2)), sigma_group1[1])T[0,1];
      a_pun[n]    ~ normal((mu[2] - (delta[2]/2)), sigma_group1[2])T[0,1];
      K[n]        ~ normal((mu[3] - (delta[3]/2)), sigma_group1[3])T[0, ];
      theta[n]    ~ normal((mu[4] - (delta[4]/2)), sigma_group1[4])T[0, ];
      omega_f[n]  ~ normal((mu[5] - (delta[5]/2)), sigma_group1[5]);
      omega_p[n]  ~ normal((mu[6] - (delta[6]/2)), sigma_group1[6]);
    }
    else if (group[n] == 2){
      a_rew[n]    ~ normal((mu[1] + (delta[1]/2)), sigma_group2[1])T[0,1];
      a_pun[n]    ~ normal((mu[2] + (delta[2]/2)), sigma_group2[2])T[0,1];
      K[n]        ~ normal((mu[3] + (delta[3]/2)), sigma_group2[3])T[0, ];
      theta[n]    ~ normal((mu[4] + (delta[4]/2)), sigma_group2[4])T[0, ];
      omega_f[n]  ~ normal((mu[5] + (delta[5]/2)), sigma_group2[5]);
      omega_p[n]  ~ normal((mu[6] + (delta[6]/2)), sigma_group2[6]);
    }
  }


  for (i in 1:N) {
    // Define values
    vector[4] ef;
    vector[4] ev;
    vector[4] PEfreq_fic;
    vector[4] PEval_fic;
    vector[4] pers;
    vector[4] util;

    real PEval;
    real PEfreq;
    real efChosen;
    real evChosen;

    // Initialize values
    ef    = initV;
    ev    = initV;
    pers  = rep_vector(1, 4);
    pers /= (1 + K[i]);
    util = initV;

    for (t in 1:Tsubj[i]) {
      // softmax choice
      choice[i, t] ~ categorical( softmax(util*theta[i]));

      // Prediction error
      PEval  = outcome[i,t] - ev[ choice[i,t]];
      PEfreq = sign_out[i,t] - ef[ choice[i,t]];
      PEfreq_fic = -sign_out[i,t]/3 - ef;

      // store chosen deck ev
      efChosen = ef[ choice[i,t]];
      evChosen = ev[ choice[i,t]];

      if (outcome[i,t] >= 0) {
        // Update ev for all decks
        ef += a_pun[i] * PEfreq_fic;
        // Update chosendeck with stored value
        ef[ choice[i,t]] = efChosen + a_rew[i] * PEfreq;
        ev[ choice[i,t]] = evChosen + a_rew[i] * PEval;
      } else {
        // Update ev for all decks
        ef += a_rew[i] * PEfreq_fic;
        // Update chosendeck with stored value
        ef[ choice[i,t]] = efChosen + a_pun[i] * PEfreq;
        ev[ choice[i,t]] = evChosen + a_pun[i] * PEval;
      }

      // Perseverance updating
      pers[ choice[i,t] ] = 1;   // perseverance term
      pers /= (1 + K[i]);        // decay

      // Utility of expected value and perseverance times theta
      util  = ev + ef * omega_f[i] + pers * omega_p[i];
    }
  }
}

generated quantities {

  // For log likelihood calculation
  array[N] real log_lik;

  // For posterior predictive check
  array[N,T] real y_pred;

  // Set all posterior predictions to -1 (avoids NULL values)
  for (i in 1:N) {
    for (t in 1:T) {
      y_pred[i,t] = -1;
    }
  }

  { // local section, this saves time and space
    for (i in 1:N) {
      // Define values
      vector[4] ef;
      vector[4] ev;
      vector[4] PEfreq_fic;
      vector[4] PEval_fic;
      vector[4] pers;   // perseverance
      vector[4] util;

      real PEval;
      real PEfreq;
      real efChosen;
      real evChosen;

      // Initialize values
      log_lik[i] = 0;
      ef    = initV;
      ev    = initV;
      pers  = rep_vector(1, 4);
      pers /= (1 + K[i]);
      util = initV;

      for (t in 1:Tsubj[i]) {
        // softmax choice
        log_lik[i] += categorical_lpmf( choice[i, t] | softmax(util*theta[i]) );

        // generate posterior prediction for current trial
        y_pred[i,t] = categorical_rng(softmax(util*theta[i]) );

        // Prediction error
        PEval  = outcome[i,t] - ev[ choice[i,t]];
        PEfreq = sign_out[i,t] - ef[ choice[i,t]];
        PEfreq_fic = -sign_out[i,t]/3 - ef;

        // store chosen deck ev
        efChosen = ef[ choice[i,t]];
        evChosen = ev[ choice[i,t]];

        if (outcome[i,t] >= 0) {
          // Update ev for all decks
          ef += a_pun[i] * PEfreq_fic;
          // Update chosendeck with stored value
          ef[ choice[i,t]] = efChosen + a_rew[i] * PEfreq;
          ev[ choice[i,t]] = evChosen + a_rew[i] * PEval;
        } else {
          // Update ev for all decks
          ef += a_rew[i] * PEfreq_fic;
          // Update chosendeck with stored value
          ef[ choice[i,t]] = efChosen + a_pun[i] * PEfreq;
          ev[ choice[i,t]] = evChosen + a_pun[i] * PEval;
        }

        // Perseverance updating
        pers[ choice[i,t] ] = 1;    // perseverance term
        pers /= (1 + K[i]);            // decay

        // Utility of expected value and perseverance
        util  = ev + ef * omega_f[i] + pers * omega_p[i];
      }
    }
  }
}
