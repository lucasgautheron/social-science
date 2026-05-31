data {
    int N;
    int M;
    array [N,M] int n;
}

parameters {
    vector<lower=0>[M] alphas;
    array[N] simplex[M] p;
    real<lower=0> scale;
}

model {
    for (i in 1:N) {
        p[i] ~ dirichlet(alphas);
        n[i] ~ multinomial(p[i]);
    }
    alphas ~ cauchy(0, 2.5);
    scale ~ cauchy(0, 2.5);
}


