data {
  int<lower=1> T; // number of trials
  array[T] int choice;
  array[T] real outcome;
  array[T] real sign_out;
}
transformed data {
  vector[4] initV;
  initV  = rep_vector(0.0, 4);
}
parameters {
  // Subject-level raw parameters (for Matt trick)
  real a_rew_pr;
  real a_pun_pr;
  real K_pr;
  real omega_f_pr;
  real omega_p_pr;
}

transformed parameters{
  // Subject-level parameters (for Matt trick)
  real<lower=0, upper=1> a_rew;
  real<lower=0, upper=1> a_pun;
  real<lower=0>          K;
  real                   omega_f;
  real                   omega_p;

  a_rew = Phi_approx(a_rew_pr);
  a_pun = Phi_approx(a_pun_pr);
  K = Phi_approx(K_pr)*5;
  omega_f = omega_f_pr;
  omega_p = omega_p_pr;
}

model {
  // individual parameters
  a_rew_pr  ~ normal(0, 1);
  a_pun_pr  ~ normal(0, 1);
  K_pr     ~ normal(0, 1);
  omega_f_pr ~ normal(0, 1);
  omega_p_pr ~ normal(0, 1);


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
    //real K_tr;

    // Initialize values
    ef    = initV;
    ev    = initV;
    pers  = rep_vector(1, 4);
    pers /= (1 + K);
    util = softmax(initV);
    //K_tr = pow(3, K) - 1;

    for (t in 1:T) {
      choice[t] ~ categorical(util);
      // Prediction error
      PEval  = outcome[t] - ev[ choice[t]];
      PEfreq = sign_out[t] - ef[ choice[t]];
      PEfreq_fic = -sign_out[t]/3 - ef;

      // store chosen deck ev
      efChosen = ef[ choice[t]];
      evChosen = ev[ choice[t]];

      if (outcome[t] >= 0) {
        // Update ev for all decks
        ef += a_pun * PEfreq_fic;
        // Update chosendeck with stored value
        ef[ choice[t]] = efChosen + a_rew * PEfreq;
        ev[ choice[t]] = evChosen + a_rew * PEval;
      } else {
        // Update ev for all decks
        ef += a_rew * PEfreq_fic;
        // Update chosendeck with stored value
        ef[ choice[t]] = efChosen + a_pun * PEfreq;
        ev[ choice[t]] = evChosen + a_pun * PEval;
      }

      // Perseverance updating
      pers[ choice[t] ] = 1;   // perseverance term
      pers /= (1 + K);        // decay

      // Utility of expected value and perseverance
      util  = softmax((ev + ef * omega_f + pers * omega_p));
    }
}


generated quantities {

  // For log likelihood calculation
  real log_lik;

  // For posterior predictive check
  array[T] real y_pred;

  // Set all posterior predictions to -1 (avoids NULL values)
  for (t in 1:T) {
    y_pred[t] = -1;
  }

  { // local section, this saves time and space

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
      //real K_tr;

      // Initialize values
      ef    = initV;
      ev    = initV;
      pers  = initV; // initial pers values
      util  = softmax(initV);
      //K_tr = pow(3, K) - 1;
      log_lik = 0;

      for (t in 1:T) {
        // softmax choice
        log_lik += categorical_lpmf( choice[t] | util );

        // generate posterior prediction for current trial
        y_pred[t] = categorical_rng(util);

        // Prediction error
        PEval  = outcome[t] - ev[ choice[t]];
        PEfreq = sign_out[t] - ef[ choice[t]];
        PEfreq_fic = -sign_out[t]/3 - ef;

        // store chosen deck ev
        efChosen = ef[ choice[t]];
        evChosen = ev[ choice[t]];

        if (outcome[t] >= 0) {
          // Update ev for all decks
          ef += a_pun * PEfreq_fic;
          // Update chosendeck with stored value
          ef[ choice[t]] = efChosen + a_rew * PEfreq;
          ev[ choice[t]] = evChosen + a_rew * PEval;
        } else {
          // Update ev for all decks
          ef += a_rew * PEfreq_fic;
          // Update chosendeck with stored value
          ef[ choice[t]] = efChosen + a_pun * PEfreq;
          ev[ choice[t]] = evChosen + a_pun * PEval;
        }

        // Perseverance updating
        pers[ choice[t] ] = 1;   // perseverance term
        pers /= (1 + K);        // decay

        // Utility of expected value and perseverance
        util  = softmax((ev + ef * omega_f + pers * omega_p));
      }
    }
  }
