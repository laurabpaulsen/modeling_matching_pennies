data {
    int<lower=1> T; // number of trials
    array[T] int choices;
    array[T] int outcomes;

    // for priors
    real prior_sd_lr;
    real prior_sd_tau;
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
    value = 0.5;
    real p;
    real diff;
    
    array[2] real utils;

    for(t in 1:T-1){
        // softmax
        p = 1 / (1 + exp(-tau * (value - 0.5))); // minus 0.5 to center around 0.5 (equal value for each hand) 

        choices[t] ~ bernoulli(p);

        // update value
        if (choices[t] == 1){
            value = value + lr * (outcomes[t] - value);
        } else {
            value = value - lr * (outcomes[t] - (1 - value));
        }
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
    array[T] int<lower=0, upper=1> pred_choice;
    array[T+1] real<lower=0, upper=1> value;
    array[2] real utils;
    real p;

    // set initial values
    value[1] = 0.5;

    // predict choice for remaining trials
    for (t in 1:T){
        utils = {value[t], 1 - value[t]};

        // softmax
        p = 1 / (1 + exp(-tau * (value[t] - 0.5))); 
        pred_choice[t] = bernoulli_rng(p);

        // update value
        if (choices[t] == 1){ // ASK RICCARDO IF WE WANT TO USE THE PREDICTED CHOICE i think not but make sure and talk about what the intuition is
            value[t+1] = value[t] + lr * (outcomes[t] - value[t]);
        } else {
            value[t+1] = value[t] - lr * (outcomes[t] - (1 - value[t]));        
        }
    }    
}