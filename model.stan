data {
    int<lower=1> T; // number of trials

    array[T] int choices;
    array[T] int outcomes;

    // for priors
    real prior_sd_lr;
    real prior_sd_tau;

}

transformed data {
    real<lower=0, upper=1> initV;
    initV = 0.5;
}


parameters {
    real loglr;
    real exptau;
}

transformed parameters {
    real<lower=0, upper=1> lr;
    real<lower=0> tau;
    tau = exp(exptau);
    lr = inv_logit(loglr);
}

model {
    // set priors
    loglr ~ normal(0, prior_sd_lr);
    exptau ~ normal(0, prior_sd_tau);

    real value;
    value = initV;
    real PE;
    real p;

    for(t in 2:T){

        // calculating the prediction error
        if (choices[t-1] == 0) {
            PE = 1 - outcomes[t-1] - value;

        } else {
            PE = outcomes[t-1] - value;
        }

        // value update
        value = value + lr * PE;

        // adding the log likelihood
        choices[t] ~ bernoulli(inv_logit(value/tau));

    }
}


generated quantities {
    real <lower=0, upper=1> prior_lr;
    real  <lower=0> prior_tau;
    
    real <lower=0, upper=1> posterior_lr;
    real  <lower=0> posterior_tau;
    
    // generate priors
    prior_lr = inv_logit(normal_rng(0, prior_sd_lr));
    prior_tau = exp(normal_rng(0, prior_sd_tau));

    // generate posteriors
    posterior_lr = inv_logit(loglr);
    posterior_tau = exp(exptau);



    // for posterior predictive checks (choices)
    array[T] real ppc_choices;

    real value;
    value = initV;
    real PE;
    real p;

    for(t in 2:T){

        // calculating the prediction error
        if (choices[t-1] == 0) {
            PE = 1 - outcomes[t-1] - value;

        } else {
            PE = outcomes[t-1] - value;
        }

        // value update
        value = value + lr * PE;

        // calculating the probability of the choice
        p = inv_logit(value/tau);

        // adding the log likelihood
        ppc_choices[t] = bernoulli_rng(p);
    }
}