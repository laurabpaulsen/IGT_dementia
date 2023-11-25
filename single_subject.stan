// Modified from https://github.com/CCS-Lab/hBayesDM/blob/534907d9cbe7101b3109feeec808cdbd380a93f2/commons/stan_files/igt_orl.stan#L4

data {
  int<lower=1> T; // number of trials
  int choice[T];
  real outcome[T];
  real sign_out[T];
}
transformed data {
  vector[4] initV;
  initV  = rep_vector(0.0, 4);
}
parameters {
  real <lower=0, upper=1> a_rew;
  real <lower=0, upper=1> a_pun;
  real K;
  real omega_f;
  real omega_p;
}

model {
  // individual parameters
  a_rew  ~ normal(0, 1.0);
  a_pun  ~ normal(0, 1.0);
  K     ~ normal(0, 1.0);
  omega_f ~ normal(0, 1.0);
  omega_p ~ normal(0, 1.0);

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
  real K_tr;

  // Initialize values
  ef    = initV;
  ev    = initV;
  pers  = initV; // initial pers values
  util  = initV;
  K_tr = pow(3, K) - 1;

  for (t in 1:T) {
      // softmax choice
      choice[t] ~ categorical_logit( util );

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
        ef[choice[t]] = efChosen + a_pun * PEfreq;
        ev[choice[t]] = evChosen + a_pun * PEval;
      }

      // Perseverance updating
      pers[ choice[t] ] = 1;   // perseverance term
      pers /= (1 + K_tr);        // decay

      // Utility of expected value and perseverance
      util  = ev + ef * omega_f + pers * omega_p;
    }
}

