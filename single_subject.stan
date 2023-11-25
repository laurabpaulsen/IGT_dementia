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
  real <lower=0, upper=1> Arew_pr;
  real <lower=0, upper=1> Apun_pr;
  real K;
  real betaF;
  real betaP;
}

model {
  // individual parameters
  Arew  ~ normal(0, 1.0);
  Apun  ~ normal(0, 1.0);
  K     ~ normal(0, 1.0);
  betaF ~ normal(0, 1.0);
  betaP ~ normal(0, 1.0);

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
    real K_tr;

    // Initialize values
    ef    = initV;
    ev    = initV;
    pers  = initV; // initial pers values
    util  = initV;
    K_tr = pow(3, K[i]) - 1;

    for (t in 1:T) {
      // softmax choice
      choice[i, t] ~ categorical_logit( util );

      // Prediction error
      PEval  = outcome[t] - ev[ choice[t]];
      PEfreq = sign_out[t] - ef[ choice[t]];
      PEfreq_fic = -sign_out[t]/3 - ef;

      // store chosen deck ev
      efChosen = ef[ choice[i,t]];
      evChosen = ev[ choice[i,t]];

      if (outcome[i,t] >= 0) {
        // Update ev for all decks
        ef += Apun[i] * PEfreq_fic;
        // Update chosendeck with stored value
        ef[ choice[t]] = efChosen + Arew * PEfreq;
        ev[ choice[t]] = evChosen + Arew * PEval;
      } else {
        // Update ev for all decks
        ef += Arew[i] * PEfreq_fic;
        // Update chosendeck with stored value
        ef[choice[t]] = efChosen + Apun * PEfreq;
        ev[choice[t]] = evChosen + Apun * PEval;
      }

      // Perseverance updating
      pers[ choice[t] ] = 1;   // perseverance term
      pers /= (1 + K_tr);        // decay

      // Utility of expected value and perseverance
      util  = ev + ef * betaF + pers * betaP;
    }
  }
}
