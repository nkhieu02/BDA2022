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

   array[n] int ypred;

   for (i in 1:n) {
    ypred[i] = binomial_rng(x[i], theta);
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
   array[g,n] int ypred;
   for (i in 1:g) {
    for (j in 1:n) {
        ypred[i, j] = binomial_rng(x[i,j], theta[i]);
    }
   }
}

