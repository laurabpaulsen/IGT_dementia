
data {
  int<lower=1> N; // number of subjecs
  int<lower=1> T; // maxiumum number of trials
  array[N] int<lower=1, upper=T> Tsubj; // trials per subject
  array[N, T] int choice; // choices made
  array[N, T] real outcome; // oucomes
  array[N, T] real sign_out; // sign of outcome
}

transformed data {
  vector[4] initV;
  initV  = rep_vector(0.0, 4);
}

parameters {
// Declare all parameters as vectors for vectorizing
  // Hyper(group)-parameters
  vector[5] mu; // overall mean

  //group level parameters
  vector<lower=0>[5] sigma;

  // Subject-level raw parameters (for Matt trick)
  vector[N] a_rew_pr;
  vector[N] a_pun_pr;
  vector[N] K_pr;
  vector[N] omega_f_pr;
  vector[N] omega_p_pr;
}

transformed parameters{
  // Subject-level parameters 
  vector<lower=0, upper=1>[N]   a_rew;
  vector<lower=0, upper=1>[N]   a_pun;
  vector<lower=0>[N]            K;
  vector[N]                     omega_f;
  vector[N]                     omega_p;


  // modelling each parameter for each subject according to the overall mean and the difference between the groups (with a group level standard deviation)
  for(n in 1:N){
      a_rew[n]    = Phi_approx(mu[1] + sigma[1] * a_rew_pr[n]);
      a_pun[n]    = Phi_approx(mu[2] + sigma[2] * a_pun_pr[n]);
      K[n]        = Phi_approx(mu[3]+ sigma[3] * K_pr[n])*5;
      omega_f[n]  = mu[4] + sigma[4] * omega_f_pr[n];
      omega_p[n]  = mu[5] + sigma[5] * omega_p_pr[n];
  }
}
model {
  // Hyperparameters
  mu ~ normal(0, 1);

  // group level parameters
  sigma[1:3] ~ normal(0, 0.2);
  sigma[4:5] ~ cauchy(0, 1.0);


  // individual parameters
  a_rew_pr  ~ normal(0, 1.0);
  a_pun_pr  ~ normal(0, 1.0);
  K_pr     ~ normal(0, 1.0);
  omega_f_pr ~ normal(0, 1.0);
  omega_p_pr ~ normal(0, 1.0);

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
      choice[i, t] ~ categorical( softmax(util));

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

      // Utility of expected value and perseverance
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
        log_lik[i] += categorical_lpmf( choice[i, t] | softmax(util) );

        // generate posterior prediction for current trial
        y_pred[i,t] = categorical_rng(softmax(util) );

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
