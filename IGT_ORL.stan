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
}

parameters {
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
  matrix[C] Ev; // Value
  matrix[C] Ef; // Frequency
  matrix[C] PS; // Perseverence
  matrix[C] p;  // probability of choice
  //vector<lower=0,upper=1>[N] a_rew;
  //vector<lower=0,upper=1>[N] a_pun;
  //vector<lower=0, upper=5>[N] K;
  //vector[N] omega_f;
  //vector[N] omega_p;


  //K = inv_logit(K_subj[subject]) * 5; // restriciting K to be between 0 and 5
  //omega_f = omega_f_subj[subject];
  //omega_p = omega_p_subj[subject];
  //a_rew = inv_logit(a_rew_subj[subject]);
  //a_pun = inv_logit(a_pun_subj[subject]);

  for (i in 1:N) {
    if (trial[i] == 1) {
      // initial values at trial 1
      for (deck in 1:C) {
        Ev[deck] = 0;
        Ef[deck] = 0;
        PS[deck] = 1;
        p [deck] = 0.25;

        current_subject = subject[i]
      }
    } else {
      for (deck in 1:C) {
        if (deck == choice[i-1]) {
          // chosen deck
          PS[deck, i] = 1.0 / (1 + K[subject[i]]);
          if (outcome[i-1] >= 0) {
            // positive, outcome
            Ev[deck] = Ev[deck] + (a_rew_subj[current_subject] * (outcome[i-1] - Ev[deck]));
            Ef[deck] = Ef[deck] + (a_rew_subj[current_subject] * (sign_out[i-1]) - Ef[deck]);
          } else {
            // negative, loss
            Ev[deck] = Ev[deck] + (a_pun_subj[current_subject] * (outcome[i-1] - Ev[deck]));
            Ef[deck] = Ef[deck] + (a_pun_subj[current_subject] * (sign_out[i-1]) - Ef[deck]);
          }
        } else {
          // the other, unchosen decks
          Ev[deck] = Ev[deck];
          PS[deck] = PS[deck,] / (1 + K_subj[current_subject]);
          if (outcome[i-1] >= 0) {
            // positive, outcome
            Ef[deck] = Ef[deck, i-1] + (a_rew_subj[current_subject] * ((-sign_out[i-1])/(C-1)) - Ef[deck]);
          } else {
            // negative, loss
            Ef[deck] = Ef[deck] + (a_pun_subj[current_subject] * ((-sign_out[i-1])/(C-1)) - Ef[deck]);
          }
        }
      }
      p[,i] = softmax((Ev[,i] + Ef[,i]*omega_f_subj[current_subject] + PS[,i]*omega_p_subj[current_subject]));
    }
  }
}



model {
  a_rew_sd ~ normal(0, 10);
  a_pun_sd ~ normal(0, 10);
  K_sd ~ normal(0, 10);
  omega_f_sd ~ normal(0, 10);
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
