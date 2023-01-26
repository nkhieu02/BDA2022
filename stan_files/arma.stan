data {
  int<lower=20> T;  // number of observations
  int T_exclude; // for validation
  vector[T] y;     // observation at time T
  vector[T_exclude] y_exclude;
}
parameters {
  real mu;              // mean
  real<lower=0> sigma;  // error scale
  real<lower = -1, upper = 1> theta;      // lag coefficients
}
transformed parameters {
  vector[T] epsilon;    // error terms
  epsilon[1] = y[1] - mu;
  for (t in 2:T) {
    epsilon[t] = ( y[t] - mu - theta * epsilon[t - 1]);
  }

  // Exclude error terms
  vector[T_exclude] epsilon_exclude;
  epsilon_exclude[1] = y_exclude[1] - mu - theta * epsilon[T];
  for (t in 2:T_exclude) {
    epsilon_exclude[t] = y_exclude[t] - mu - theta * epsilon_exclude[t - 1];
  }
}
model {
  mu ~ normal(0, 2);
  theta ~ normal(0, 2);
  sigma ~ cauchy(0, 2.5);
  for (t in 2:T) {
    y[t] ~ normal(mu + theta * epsilon[t - 1] , sigma);
  }
}
generated quantities {
   vector[T-1] ypred;
   vector[T-1] log_lik;
   vector[T_exclude] ypred_exclude;
   vector[T_exclude] log_lik_exclude;
   for (t in 1:T-1) {
    ypred[t] = normal_rng(mu + theta * epsilon[t] , sigma);
    log_lik[t] = normal_lpdf(y[t] | mu+ theta * epsilon[t], sigma);
   }
   for (t in 1:T_exclude) {
    ypred_exclude[t] = normal_rng(mu + theta * epsilon_exclude[t] , sigma);
    log_lik_exclude[t] = normal_lpdf(y_exclude[t] | mu+ theta * epsilon_exclude[t], sigma);
   }
}

//-----
data {
  int<lower=1> T;
  matrix[2, T] y;
}
parameters {
  vector[2] mu; // mean coeff
  matrix[2,2] phi;
  matrix[2,2] theta;
  cov_matrix[2] sigma;
}
model {
  matrix[2, T] nu;
  matrix[2, T] err;
  nu[:, 1] = mu + phi * mu;
  err[:, 1] = y[:, 1] - nu[:, 1];
  for (t in 2:T) {
    nu[:, t] = mu + phi * y[:, t-1] + theta * err[:, t-1];
    err[:, t] = y[:, t] - nu[:, t];
  }
  for (t in 1:T) {
    err[:, t] ~ multi_normal([0,0], sigma);
  }
}
