data {
    int<lower=1> T; // number of trials

    array[T] int choices;
    array[T] int outcomes;

}

transformed data {
    real<lower=0, upper=1> initV;
    initV = 0.5;
}


parameters {
    real<lower=0, upper=1> lr;
    real tau;
}

transformed parameters {
    real<lower=0> tau;
    tau = inv_logit(logTau);
}

model {
    // set priors
    target += uniform_lpdf(lr | 0, 1);
    target += normal_lpdf(logTau | 0, 1); 

    real value;
    value = initV

    for(t in 2:T){

        // calculating the prediction error
        if (choices[t-1] == 0) {
            PE = 1 - outcomes[t-1] - value;

        } else {
            PE = outcomes[t-1] - value;
        }

        // value update
        value = value + alpha * PE

        p = inv_logit(-tau * value);

        // making a prediction
        target += bernoulli_lpmf(choices[t] | p);

    }

}