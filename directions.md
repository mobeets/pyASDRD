## Planning

_2014-10-13_

__Big story (for NSF)__

* go through draft together sometime soon

__Next part of the story__

* need to pin down a good series of next steps, probably parallel ones
* don't want to get stuck, don't want to bug

__Python vs. MATLAB__

* python: less stuckness for me
* scikit-learn: open-source, Bayesian ARD, good cross-validation tools

## Methods
_2014-10-13_

__Comparing many things in parallel:__

* _ground truths_: bilinear vs. full vs. space vs. time
* _dimensions of fits_: [same as ground truths]
* _regularizers_: none vs. ridge vs. ASD vs. ARD vs. ASDRD
* _metrics_: rmse of prediction rates; % training vs. k-fold

__Bilinear and/or regularization__

* Hyperparameter fitting for each step of ALS?
* Or, focus on regularization and then assume ALS will improve fits

__Logistic regression__

* Good time to try to figure this out?

## Notes
_2014-10-14_

* scikit-learn licensing? BSD (n.b. don't use GPL)
* need to verify ALS without prior FAILS, if you want to show ASD has advantage; a) add noise, b) less data

### Full stimulus

* check out stimulus covariance matrix (X.T*X) -- should just be a diagonal or no?
* can add two bilinear matrices to get full -- not random weights, or else there's no structure for a prior to help you with!
* full rank-k matrix recipe: USV where S is kxk diagonal of weights, U is k cols of space weights, V is k rows of time weights
* sanity check: ASD should not smooth a random matrix, i.e., ASD should show no advantage over OLS when there is no structure to the weights

### This week

1. visualizer for full weights: see [Ghose paper](http://www.ghoselab.cmrr.umn.edu/Publications/19819253.pdf), Fig. 2
2. find sweet spot for # trials, amount of noise; e.g. ~500 trials per day, real noisy; want to see OLS and ALS fail; are ASD, ARD, etc. improving?
3. grid apertures, color map
