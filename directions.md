_2014-10-13_

## Planning

__Big story (for NSF)__

* go through draft
* attempt to fill in methods and then get feedback again?

__Next part of the story__

* need to pin down a good series of next steps, probably parallel ones
* don't want to get stuck, don't want to bug

__Python vs. MATLAB__

* python: less stuckness for me
* scikit-learn: open-source, Bayesian ARD, good cross-validation tools

## Methods

__Comparing many things in parallel:__

* _ground truths_: bilinear vs. full vs. space vs. time
* _dimensions of fits_: [same as ground truths]
* _regularizers_: none vs. ridge vs. ASD vs. ARD vs. ASDRD
* _metrics_: rmse of prediction rates; % training vs. k-fold

__Bilinear vs. regularization__

* Hyperparameter fitting for each step of ALS?
* Or, focus on regularization and then assume ALS will improve fits

__Logistic regression__

* Good time to try to figure this out?
