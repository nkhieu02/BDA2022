// Pool model
data {
    int<lower=0> n;
    array[n] int x;
    array[n] int y;
}
parameters {
    real<lower=0, upper=1> theta;
}
model {
    theta ~ beta(1,1);
    y ~ binomial(x, theta);
}
generated quantities {
   vector[n] log_lik;
   array[n] int ypred;

   for (i in 1:n) {
    ypred[i] = binomial_rng(x[i], theta);
    log_lik[i] = binomial_lpmf(y[i] | x[i], theta);
   }
}
//---

// Separated model
data {
    int<lower = 0> g; 
    int<lower=0> n;
    array[g,n] int x;
    array[g,n] int y;
}
parameters {
    array[g] real<lower=0, upper=1> theta;
}
model {
    for (i in 1:g) {
        theta[i] ~ beta(1,1);
        y[i] ~ binomial(x[i], theta[i]);
    }
}
generated quantities {
   array[g, n] real log_lik;
   array[g,n] int ypred;
   for (i in 1:g) {
    for (j in 1:n) {
        log_lik[i, j] = binomial_lpmf(y[i, j] | x[i, j], theta[i]);
        ypred[i, j] = binomial_rng(x[i,j], theta[i]);
    }
   }
}

