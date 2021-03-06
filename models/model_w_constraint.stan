data {
  int<lower=0> G;
  int<lower=0> N_TEAMS;
  int<lower=1, upper=N_TEAMS> h_g[G];
  int<lower=1, upper=N_TEAMS> a_g[G];
  int<lower=0> y_g1[G];
  int<lower=0> y_g2[G];
  
  int<lower=0> PRED_G;
  int<lower=1, upper=N_TEAMS> future_h_g[PRED_G];
  int<lower=1, upper=N_TEAMS> future_a_g[PRED_G];  
}

parameters {
  real home;
  
  real mu_att;
  real mu_def;
  real<lower=0> tau_att;
  real<lower=0> tau_def;
  vector[N_TEAMS] att_star;
  vector[N_TEAMS] def_star;
}

transformed parameters{
  vector[N_TEAMS] att;
  vector[N_TEAMS] def;
  for (i in 1:N_TEAMS){
    att[i] = att_star[i] - mean(att_star);
    def[i] = def_star[i] - mean(def_star);
  }
}

model {
  vector[G] theta_g1;
  vector[G] theta_g2;
  
  for (i in 1:G){
   theta_g1[i] = exp(home + att[h_g[i]] + def[a_g[i]]); 
   theta_g2[i] = exp(att[a_g[i]] + def[h_g[i]]); 
  }
  
  home ~ normal(0,0.001);
  
  mu_att ~ normal(0,0.001);
  tau_att ~ gamma(0.1, 0.1);
  mu_def ~ normal(0,0.001);
  tau_def ~ gamma(0.1, 0.1);
  
  for (i in 1:N_TEAMS){
    att_star[i] ~ normal(mu_att, tau_att);
    def_star[i] ~ normal(mu_def, tau_def);
  }
  
  y_g1 ~ poisson(theta_g1);
  y_g2 ~ poisson(theta_g2);
}

generated quantities {
  vector[G] y_g1_rep;
  vector[G] y_g2_rep;
  
  vector[PRED_G] y_g1_pred;
  vector[PRED_G] y_g2_pred;
  
  for (i in 1:G){
    real theta_g1_rep = exp(home + att[h_g[i]] + def[a_g[i]]);
    real theta_g2_rep = exp(att[a_g[i]] + def[h_g[i]]);
    
    y_g1_rep[i] = poisson_rng(theta_g1_rep);
    y_g2_rep[i] = poisson_rng(theta_g2_rep);
  }
  
  for (i in 1:PRED_G){
    real theta_g1_pred = exp(home + att[future_h_g[i]] + def[future_a_g[i]]);
    real theta_g2_pred = exp(att[future_a_g[i]] + def[future_h_g[i]]);
    
    y_g1_pred[i] = poisson_rng(theta_g1_pred);
    y_g2_pred[i] = poisson_rng(theta_g2_pred);
  }
}