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
  
  real mu_att[3];
  real mu_def[3];
  real<lower=0> tau_att[3];
  real<lower=0> tau_def[3];
  vector[N_TEAMS] att_star;
  vector[N_TEAMS] def_star;
  matrix[N_TEAMS, 3] p_att;
  matrix[N_TEAMS, 3] p_def;
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
  
  #bottom table teams
  mu.att[1] ~ normal(0,0.001) T[-3,0];
  mu.def[1] ~ normal(0,0.001) T[0,3];
  tau.att[1] ~ gamma(0.01,0.01);
  tau.def[1] ~ gamma(0.01,0.01);
  
  #middle table teams
  mu_att[1] ~ normal(0,0);
  mu_def[1] ~ normal(0,0);
  tau_att[1] ~ gamma(0.01,0.01);
  tau_def[1] ~ gamma(0.01,0.01);
  
  #upper table teams
  mu_att[3] ~ normal(0,0.001) T[0,3];
  mu_def[3] ~ normal(0,0.001) T[-3,0];
  tau_att[3] ~ gamma(0.01,0.01);
  tau_def[3] ~ gamma(0.01,0.01);
  
  for (i in 1:N_TEAMS){
    grp_att[i] ~ categorical(p_att[i,]);
    grp_def[i] ~ categorical(p_def[i,]);
    att_star[t] ~ student_t(grp_att[i], tau_att[grp_att[i]], 4)
    def_star[t] ~ student_t(grp_def[i], tau_def[grp_def[i]], 4)
    
    att_star[i] ~ normal(mu_att, tau_att);
    def_star[i] ~ normal(mu_def, tau_def);
    
    p_att[i, 1:3] ~ dirichlet()
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