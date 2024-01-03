data {
  int<lower=1> N;
  int<lower=1> T;
  array[N] int<lower=1, upper=T> Tsubj;
  array[N, T] int choice;
  array[N, T] real outcome;
  array[N, T] real sign_out;
}
transformed data {
  vector[4] initV;
  initV  = rep_vector(0.0, 4);
}
parameters {
// Declare all parameters as vectors for vectorizing
  // Hyper(group)-parameters
  vector[6] mu_pr;
  vector<lower=0>[6] sigma;

  // Subject-level raw parameters (for Matt trick)
  vector[N] a_rew_pr;
  vector[N] a_pun_pr;
  vector[N] K_pr;
  vector[N] omega_f_pr;
  vector[N] omega_p_pr;
  vector[N] theta_pr;
}
transformed parameters {
  // Transform subject-level raw parameters
  vector<lower=0, upper=1>[N] a_rew;
  vector<lower=0, upper=1>[N] a_pun;
  vector<lower=0, upper=5>[N] K;
  vector<lower=0, upper=5>[N] theta;
  vector[N]                   omega_f;
  vector[N]                   omega_p;


  for (i in 1:N) {
    a_rew[i] = Phi_approx(mu_pr[1] + sigma[1] * a_rew_pr[i]);
    a_pun[i] = Phi_approx(mu_pr[2] + sigma[2] * a_pun_pr[i]);
    K[i]    = Phi_approx(mu_pr[3] + sigma[3] * K_pr[i]) * 5;
    theta[i]    = Phi_approx(mu_pr[4] + sigma[4] * theta_pr[i]) * 5;
  }
  omega_f = mu_pr[4] + sigma[5] * omega_f_pr;
  omega_p = mu_pr[5] + sigma[6] * omega_p_pr;
}
model {
  // Hyperparameters
  mu_pr  ~ normal(0, 1);
  sigma[1:4] ~ normal(0, 0.2);
  sigma[5:6] ~ cauchy(0, 1.0);

  // individual parameters
  a_rew_pr  ~ normal(0, 1.0);
  a_pun_pr  ~ normal(0, 1.0);
  K_pr     ~ normal(0, 1.0);
  omega_f_pr ~ normal(0, 1.0);
  omega_p_pr ~ normal(0, 1.0);
  theta ~ normal(0, 1.0);

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

    for (t in 1:Tsubj[i]) {
      // softmax choice
      choice[i, t] ~ categorical_logit( util*theta[i] );

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
      pers /= (1 + K_tr);        // decay

      // Utility of expected value and perseverance
      util  = ev + ef * omega_f[i] + pers * omega_p[i];
    }
  }
}

generated quantities {
  // For group level parameters
  real<lower=0,upper=1> mu_a_rew;
  real<lower=0,upper=1> mu_a_pun;
  real<lower=0,upper=5> mu_K;
  real<lower=0,upper=5> mu_theta;
  real                  mu_omega_f;
  real                  mu_omega_p;

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

  mu_a_rew    = Phi_approx(mu_pr[1]);
  mu_a_pun    = Phi_approx(mu_pr[2]);
  mu_K        = Phi_approx(mu_pr[3]) * 5;
  mu_theta    = Phi_approx(mu_pr[4]) * 5;
  mu_omega_f  = mu_pr[5];
  mu_omega_p  = mu_pr[6];

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
      real K_tr;

      // Initialize values
      log_lik[i] = 0;
      ef    = initV;
      ev    = initV;
      pers  = initV; // initial pers values
      util  = initV;
      K_tr = pow(3, K[i]) - 1;

      for (t in 1:Tsubj[i]) {
        // softmax choice
        log_lik[i] += categorical_logit_lpmf( choice[i, t] | util*theta[i] );

        // generate posterior prediction for current trial
        y_pred[i,t] = categorical_rng(softmax(util*theta[i]));

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
        pers /= (1 + K_tr);        // decay

        // Utility of expected value and perseverance
        util  = ev + ef * omega_f[i] + pers * omega_p[i];
      }
    }
  }
}