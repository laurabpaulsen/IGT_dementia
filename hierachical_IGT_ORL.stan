data {
  int N; // number of trials (total for all subjects)
  int C; // number of decks, often 4 for IGT
  int Nsubj; // how many subjects
  // choice and outcome
  array[N] int trial;
  array[N] int subject;
  array[N] int choice;
  array[N] real outcome;
  array[N] real sign_out;

  int N_beta;
  matrix[N, N_beta] X;
}

parameters {
  // learning rates
  // hierarchical
  vector[N_beta] a_rew_betas;
  vector[N_beta] a_pun_betas;
  vector[N_beta] K_betas;
  vector[N_beta] omega_p_betas;
  vector[N_beta] omega_f_betas;

  vector<lower=0,upper=1>[Nsubj] a_pun_subj;
  vector<lower=0,upper=1>[Nsubj] a_rew_subj;
  vector<lower=0>[Nsubj] K_subj;
  vector[Nsubj] omega_f_subj;
  vector[Nsubj] omega_p_subj;

  real<lower=0> a_rew_sd;
  real<lower=0> a_pun_sd;
  real<lower=0> K_sd;
  real<lower=0> omega_p_sd;
  real<lower=0> omega_f_sd;

}

transformed parameters {
  matrix[C,N] Ev; // Value
  matrix[C,N] Ef; // Frequency
  matrix[C,N] PS; // Perseverence
  matrix[C,N] p;  // probability of choice
  vector<lower=0,upper=1>[N] a_rew;
  vector<lower=0,upper=1>[N] a_pun;
  vector<lower=0>[N] K;
  vector[N] omega_f;
  vector[N] omega_p;


  K = inv_logit(X * K_betas +  K_subj[subject]); // restriciting K to be between 0 and 5
  omega_f = X * omega_f_betas + omega_f_subj[subject];
  omega_p = X * omega_p_betas + omega_p_subj[subject];
  a_rew = inv_logit(X * a_rew_betas + a_rew_subj[subject]);
  a_pun = inv_logit(X * a_pun_betas + a_pun_subj[subject]);

  for (i in 1:N) {
    if (trial[i] == 1) {
      // initial values at trial 1
      for (deck in 1:C) {
        Ev[deck,i] = 0;
        Ef[deck,i] = 0;
        PS[deck,i] = 1;
        p [deck,i] = 0.25;
      }
    } else {
      for (deck in 1:C) {
        if (deck == choice[i-1]) {
          // chosen deck
          PS[deck, i] = 1.0 / (1 + K[subject[i]]);
          if (outcome[i-1] >= 0) {
            // positive, outcome
            Ev[deck, i] = Ev[deck, i-1] + (a_rew[i] * (outcome[i-1] - Ev[deck, i-1]));
            Ef[deck, i] = Ef[deck, i-1] + (a_rew[i] * (sign_out[i-1]) - Ef[deck, i-1]);
          } else {
            // negative, loss
            Ev[deck, i] = Ev[deck, i-1] + (a_pun[i] * (outcome[i-1] - Ev[deck, i-1]));
            Ef[deck, i] = Ef[deck, i-1] + (a_pun[i] * (sign_out[i-1]) - Ef[deck, i-1]);
          }
        } else {
          // the other, unchosen decks
          Ev[deck, i] = Ev[deck, i-1];
          PS[deck, i] = PS[deck, i-1] / (1 + K[i]);
          if (outcome[i-1] >= 0) {
            // positive, outcome
            Ef[deck, i] = Ef[deck, i-1] + (a_rew[i] * ((-sign_out[i-1])/(C-1)) - Ef[deck, i-1]);
          } else {
            // negative, loss
            Ef[deck, i] = Ef[deck, i-1] + (a_pun[i] * ((-sign_out[i-1])/(C-1)) - Ef[deck, i-1]);
          }
        }
      }
      p[,i] = softmax((Ev[,i] + Ef[,i]*omega_f[i] + PS[,i]*omega_p[i]));
    }
  }
}



model {
  a_rew_betas ~ normal(0, 10);
  a_rew_sd ~ normal(0, 10);

  a_pun_betas ~ normal(0, 10);
  a_pun_sd ~ normal(0, 10);

  K_betas ~ normal(0, 10);
  K_sd ~ normal(0, 10);

  omega_f_betas ~ normal(0, 10);
  omega_f_sd ~ normal(0, 10);

  omega_p_betas ~ normal(0, 10);
  omega_p_sd ~ normal(0, 10);

  a_rew_subj ~ normal(0, a_rew_sd);
  a_pun_subj   ~ normal(0, a_pun_sd);
  K_subj       ~ normal(0, K_sd);
  omega_f_subj ~ normal(0, omega_f_sd);
  omega_p_subj ~ normal(0, omega_p_sd);

  for (t in 2:N) {
    choice[t] ~ categorical(p[,t]);
  }
}
