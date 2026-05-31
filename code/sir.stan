functions {
  array[] real sir(real t, array[] real y, array[] real theta,
             array [] real x_r, array[] int x_i) {

      real S = y[1];
      real I = y[2];
      real R = y[3];
      real N = x_i[1];
      
      real beta = theta[1];
      real gamma = theta[2];
      
      real dS_dt = -beta * I * S / N;
      real dI_dt =  beta * I * S / N - gamma * I;
      real dR_dt =  gamma * I;
      
      return {dS_dt, dI_dt, dR_dt};
  }
}
data {
  int<lower=1> n_years;
  array [n_years] int total;
  array [n_years] int cases;
}

parameters {
    vector[n_years] beta;
    real<lower=0> sd;
}

model {
    vector [n_years] p = inv_logit(beta);
    cases ~ binomial(total, p);
    beta[1] ~ normal(0, 1);

    vector[n_years-1] ll;
    
    for (t in 2:n_years) {
        target += normal_lpdf(beta[t] | beta[t-1], sd);
    }

    sd ~ cauchy(0, 2.5);
}

generated quantities {
    vector[n_years] delta;
    delta[1] = 0;
    for (t in 2:n_years) {
        delta[t] = beta[t]-beta[t-1];
    }
}