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
    target += normal_lpdf(lr | 0, prior_sd_lr);
    target += normal_lpdf(tau | 0, prior_sd_tau);

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
        target += bernoulli_lpmf(choices[t] | p);

    }
}


generated quantities {

    // generate priors
    real prior_lr = normal_rng(0, prior_sd_lr);
    real prior_tau = normal_rng(0, prior_sd_tau);


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