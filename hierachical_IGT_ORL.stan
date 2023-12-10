
data {
  int<lower=1> N; // number of subjecs
  int<lower=1> T; // maxiumum number of trials
  int<lower=1, upper=T> Tsubj[N]; // trials per subject
  int choice[N, T]; // choices made
  real outcome[N, T]; // oucomes
  real sign_out[N, T]; // sign of outcome
  int N_beta; // number of model coefficients
  matrix[N, N_beta] X; // design matrix
}
transformed data {
  vector[4] initV;
  initV  = rep_vector(0.0, 4);
}
parameters {
// Declare all parameters as vectors for vectorizing
  // Hyper(group)-parameters
  matrix[N_beta, 6] beta_p;
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
  vector<lower=0, upper=1>[N]   a_rew;
  vector<lower=0, upper=1>[N]   a_pun;
  vector<lower=0, upper=5>[N]   K;
  vector<lower=0, upper=10>[N]  theta;
  vector[N]                     omega_f;
  vector[N]                     omega_p;

  a_rew = Phi_approx(X * beta_p[,1] + a_rew_pr);
  a_pun = Phi_approx(X * beta_p[,2] + a_pun_pr);
  K     = Phi_approx(X * beta_p[,3] + K_pr) * 5;
  theta   = Phi_approx(X * beta_p[,4] + K_pr) * 5;

  omega_f = X * beta_p[,5] + omega_f_pr;
  omega_p = X * beta_p[,6] + omega_p_pr;

  
}
model {
  sigma[1:4] ~ normal(0, 0.2);
  sigma[5:6] ~ cauchy(0, 1.0);


  //QUESTION: SHOULD SIGMA WORK ON THE BETAS INSTEAD OF THE SUBJECT LEVEL PARAMS? the way it is setup now sigma is across both groups?
  // Hyperparameters
  for (idx in 1:6){
    beta_p[,idx]  ~ normal(0, 1);
  }



  // individual parameters
  a_rew_pr   ~ normal(0, sigma[1]);
  a_pun_pr   ~ normal(0, sigma[2]);
  K_pr       ~ normal(0, sigma[3]);
  theta_pr   ~ normal(0, sigma[4]);
  omega_f_pr ~ normal(0, sigma[5]);
  omega_p_pr ~ normal(0, sigma[6]);


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
    //real K_tr;

    // Initialize values
    ef    = initV;
    ev    = initV;
    pers  = rep_vector(1, 4);
    pers /= (1 + K[i]);
    util = softmax(initV*theta[i]);
    //K_tr = pow(3, K[i]) - 1;

    for (t in 1:Tsubj[i]) {
      // softmax choice
      choice[i, t] ~ categorical( util );

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
      util  = softmax((ev + ef * omega_f[i] + pers * omega_p[i])*theta[i]);
    }
  }
}


generated quantities {

  // For log likelihood calculation
  real log_lik[N];

  // For posterior predictive check
  real y_pred[N,T];

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
      //real K_tr;

      // Initialize values
      log_lik[i] = 0;
      ef    = initV;
      ev    = initV;
      pers  = rep_vector(1, 4);
      pers /= (1 + K[i]);
      util = softmax(initV*theta[i]);
      //K_tr = pow(3, K[i]) - 1;

      pers  = initV; // initial pers values
      util  = initV;
      //K_tr = pow(3, K[i]) - 1;

      for (t in 1:Tsubj[i]) {
        // softmax choice
        log_lik[i] += categorical_lpmf( choice[i, t] | util );

        // generate posterior prediction for current trial
        y_pred[i,t] = categorical_rng(util);

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
        util  = softmax((ev + ef * omega_f[i] + pers * omega_p[i]) * theta[i]);
      }
    }
  }
}
