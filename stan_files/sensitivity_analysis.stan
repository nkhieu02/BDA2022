data {
  int<lower=20> T;  // number of observations
  int T_exclude; // for validation
  vector[T + T_exclude] y;     // observation at time T
}
parameters {
  real mu;              // mean
  real<lower=0> sigma;  // error scale
  real<lower = -1, upper = 1> theta;      // lag coefficients
}
transformed parameters {
  vector[T+T_exclude] epsilon;    // error terms
  epsilon[1] = y[1] - mu;
  for (t in 2:(T+T_exclude)) {
    epsilon[t] = ( y[t] - mu - theta * epsilon[t - 1]);
  }
}
model {
  mu ~ normal(1, 3);
  theta ~ normal(0, 1);
  sigma ~ normal(0, 2);
  for (t in 2:T) {
    y[t] ~ normal(mu + theta * epsilon[t - 1] , sigma);
  }
}
generated quantities {
   vector[T + T_exclude - 1] ypred; // Since not include the first observation
   vector[T+T_exclude - 1] log_lik;

   for (t in 1:(T + T_exclude-1)) {
    ypred[t] = normal_rng(mu + theta * epsilon[t] , sigma);
    log_lik[t] = normal_lpdf(y[t] | mu+ theta * epsilon[t], sigma);
   }
}

//-----
data {
  int<lower=20> T;
  int T_exclude;
  // int T_exclude;
  vector[T + T_exclude] y_1;
  vector[T + T_exclude] y_2;
}
parameters {
  real mu_1; 
  real mu_2;
  real<lower = -1, upper = 1> theta_1_1;
  real<lower = -1, upper = 1> theta_1_2;
  real<lower = -1, upper = 1> theta_2_1;
  real<lower = -1, upper = 1> theta_2_2;
  real<lower = 0> sigma_1;
  real<lower = 0> sigma_2; 
}
transformed parameters {
  vector[T + T_exclude] epsilon_1;
  vector[T + T_exclude] epsilon_2;
  epsilon_1[1] = y_1[1] - mu_1;
  epsilon_2[1] = y_2[1] - mu_2;
  for (t in 2:(T + T_exclude)) {
    epsilon_1[t] = y_1[t] - mu_1 - theta_1_1 * epsilon_1[t - 1] - theta_1_2 * epsilon_2[t-1];
    epsilon_2[t] = y_2[t] - mu_2 - theta_2_1 * epsilon_1[t - 1] - theta_2_2 * epsilon_2[t-1];
  }
}
model {
  mu_1 ~ normal(1, 3);
  mu_2 ~ normal(1, 3);
  sigma_1 ~ normal(0, 2.5);
  sigma_2 ~ normal(0, 2.5);
  theta_1_1 ~ normal(0, 1);
  theta_1_2 ~ normal(0, 1);
  theta_2_1 ~ normal(0, 1);
  theta_2_2 ~ normal(0, 1);

  for (t in 2:T) {
    y_1[t] ~ normal(mu_1 + theta_1_1 * epsilon_1[t-1] + theta_1_2 * epsilon_2[t-1], sigma_1);
    y_2[t] ~ normal(mu_2 + theta_2_1 * epsilon_1[t-1] + theta_2_2 * epsilon_2[t-1], sigma_2);
  }
}
generated quantities {
  vector[T + T_exclude - 1] ypred_1;
  vector[T + T_exclude - 1] ypred_2;
  vector[T + T_exclude - 1] log_lik;

  for (t in 1:(T + T_exclude -1)) {
    ypred_1[t] = normal_rng(mu_1 + theta_1_1 * epsilon_1[t] + theta_1_2 * epsilon_2[t], sigma_1);
    ypred_2[t] = normal_rng(mu_2 + theta_2_1 * epsilon_1[t] + theta_2_2 * epsilon_2[t], sigma_2);
    log_lik[t] = normal_lpdf(y_1[t] | mu_1 + theta_1_1 * epsilon_1[t] + theta_1_2 * epsilon_2[t], sigma_1);
  }
}