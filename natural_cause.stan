// Pool model
data {
   int<lower = 0> N;
   int<lower = 0> y;
}
parameters {
   real<lower=0, upper=1> theta;
}
model {
   theta ~ beta(1,1);
   y ~ binomial(N, theta);
}

//---

// Separated model
data {
    int<lower = 0> g; 
    int<lower = 0> N[g];
    int<lower = 0> y[g];
}
parameters {
    real<lower=0, upper=1> theta[g];
}
model {
    theta ~ beta(1,1);
    y ~ binomial(N, theta);

}

//---

// Hierarchical model
data {
    int<lower = 0> g; 
    int<lower = 0> N[g];
    int<lower = 0> y[g];
}
parameters {
    real mu;
    real<lower = 0> sigma;
    real<lower=0, upper=1> theta[g];
}
model {
    mu ~ normal(0,1);
    sigma ~ inv_chi_square(1);
    theta ~ normal(mu, sigma);
    y ~ binomial(N, theta);
}
