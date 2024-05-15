Here represented derivation of transferring original discrete-in-time setup to continuous one.   
All formulas are on the slides below.

A few comments on them:  

1. Finding forward SDE, for which original markov chain is one of possible discretizations.
2. Under some mild conditions we can get reverse ODE preserving marginal distributions.
![Forward and Backward Processes](./theoretical_part/docs/1_forward_and_backward.jpg)

3. To specify forward SDE we work with, we have to define $\alpha(t)$. Rational choice here is to use function fitting discrete schedule from the paper. Proposed family of function fit shedules from paper well. The best config from paper corresponds to $alpha=0.0136$.
![Forward and Backward Processes](./theoretical_part/docs/2_alpha_t.jpg)

4. Setting a loss for training and getting different parametrization like it happened for diffusion models.
![Forward and Backward Processes](./theoretical_part/docs/3_parametrizations.jpg)
