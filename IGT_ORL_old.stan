data {
  int C; // number of decks, often 4 for IGT
  int N; // number of trials
  array[N] int choice;
  array[N] real outcome;
  array[N] real sign_out;
}

parameters {
  real <lower=0, upper=1> a_pun;
  real <lower=0, upper=1> a_rew;
  real <lower=0>  K;
  real omega_f;
  real omega_p;

}

transformed parameters {
  vector[C] Ev; // Value
  vector[C] Ef; // Frequency
  vector[C] PS; // Perseverence
  vector[C] p;  // probability of choice

  for (i in 1:N) {
    if (i == 1) {
      // initial values at trial 1
      for (deck in 1:C) {
        Ev[deck] = 0;
        Ef[deck] = 0;
        PS[deck] = 1;
        p [deck] = 0.25;
      }
    } else {
      for (deck in 1:C) {
        if (deck == choice[i-1]) {
          // chosen deck
          PS[deck] = 1.0 / (1 + K);
          if (outcome[i-1] >= 0) {
            // positive, outcome
            Ev[deck] = Ev[deck] + (a_rew * (outcome[i-1] - Ev[deck]));
            Ef[deck] = Ef[deck] + (a_rew * (sign_out[i-1]) - Ef[deck]);
          } else {
            // negative, loss
            Ev[deck] = Ev[deck] + (a_pun * (outcome[i-1] - Ev[deck]));
            Ef[deck] = Ef[deck] + (a_pun* (sign_out[i-1]) - Ef[deck]);
          }
        } else {
          // the other, unchosen decks
          PS[deck] = PS[deck] / (1 + K);
          if (outcome[i-1] >= 0) {
            // positive, outcome
            Ef[deck] = Ef[deck] + (a_rew * ((-sign_out[i-1])/(C-1)) - Ef[deck]);
          } else {
            // negative, loss
            Ef[deck] = Ef[deck] + (a_pun * ((-sign_out[i-1])/(C-1)) - Ef[deck]);
          }
        }
      }
      p = softmax((Ev + Ef*omega_f + PS*omega_p));
    }
  }
}



model {
  a_rew ~ uniform(0, 1);
  a_pun   ~ uniform(0, 1);
  K       ~ normal(0, 5) T[0,];
  omega_f ~ normal(0, 1);
  omega_p ~ normal(0, 1);

  //a_rew ~ normal(0, 1);
  //a_pun   ~ normal(0, 1);
  //K       ~ normal(0, 1);
  //omega_f ~ normal(0, 1);
  //omega_p ~ normal(0, 1);

  for (t in 2:N) {
    choice[t] ~ categorical(p);
  }
}
