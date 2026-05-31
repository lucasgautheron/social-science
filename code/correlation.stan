data {
  int<lower=1> N;                    // number of observations
  int<lower=2> K;                    // number of categories
  array[N,K] int<lower=0> y;        // multinomial outcomes (N x K matrix)
  array[N] int<lower=0> n;                 // total trials for each observation
}

parameters {
  vector[K-1] mu;                    // mean of beta (K-1 dimensional, last category as reference)
  matrix[K-1, N] beta_raw;           // raw beta parameters (before transformation)
  cholesky_factor_corr[K-1] L_Omega; // Cholesky factor of correlation matrix
  vector<lower=0>[K-1] tau;          // scale parameters
}

transformed parameters {
  matrix[K-1, K-1] L_Sigma;          // Cholesky factor of covariance matrix
  matrix[N, K-1] beta;               // transformed beta parameters
  matrix[N, K] beta_full;            // full beta matrix (including reference category)
  array[N] simplex[K] p;                    // probabilities from softmax

  // Construct Cholesky factor of covariance matrix
  L_Sigma = diag_pre_multiply(tau, L_Omega);

  // Transform raw parameters to get correlated betas
  for (i in 1:N) {
    beta[i] = (mu + L_Sigma * beta_raw[:, i])';
  }

  // Add reference category (set to 0)
  beta_full = append_col(beta, rep_vector(0, N));

  // Apply softmax to get probabilities
  for (i in 1:N) {
    p[i] = softmax(beta_full[i]');
  }
}

model {
  // Priors
  mu ~ normal(0, 2);                 // prior on mean
  tau ~ exponential(1);              // prior on scale parameters
  L_Omega ~ lkj_corr_cholesky(2);    // LKJ prior on correlation matrix

  // Prior on raw parameters (standard normal)
  to_vector(beta_raw) ~ normal(0, 1);

  // Likelihood
  for (i in 1:N) {
    y[i] ~ multinomial(to_vector(p[i]));
  }
}

generated quantities {
  matrix[K-1, K-1] Omega;            // correlation matrix
  matrix[K-1, K-1] Sigma;            // covariance matrix

  // Recover correlation and covariance matrices
  Omega = multiply_lower_tri_self_transpose(L_Omega);
  Sigma = multiply_lower_tri_self_transpose(L_Sigma);
}