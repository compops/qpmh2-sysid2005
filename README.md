# qpmh2-sysid2015

This code was downloaded from < https://github.com/compops/qpmh2-sysid2015 > or from < http://liu.johandahlin.com/ > and contains the code used to produce the results in the paper

J. Dahlin, F. Lindsten and T. B. Schön, **Quasi-Newton particle Metropolis-Hastings**. Proceedings of the 17th IFAC Symposium on System Identification, Beijing, China, October 2015. 

The papers are available as a preprint from < http://arxiv.org/pdf/1502.03656 > and < http://liu.johandahlin.com/ >.

Requirements
--------------
The code is written and tested for Python 2.7.6. The implementation makes use of NumPy 1.9.2, SciPy 0.15.1, Matplotlib 1.4.3 and Pandas 0.13.1. On Ubuntu, these packages can be installed/upgraded using "sudo pip install --upgrade package-name ".

Included files
--------------
**RUNME-kalman.py**
Compares the pPMH0, pPMH1 and qPMH2 algorithms using the Kalman smoother to estimate the required gradient and log-likelihood. This is an simplified example of the LGSS model in the paper that runs fairly quickly. The code presents the trace plots, the resulting posterior estimates and computes an approximate value of the IACT (IF in the paper) for the different proposals and parameters. Note that the IACT calculation is carried out in R for the paper but the code is written to resemble the R implementation as close as possible. The plots and data from a run of this file is attached in the results folder.

**RUNME-particles.py**
Makes the same comparison as in *RUNME-kalman.py* but by using the fully adapted particle filter instead as done in the paper. This code takes a bit longer to run but generates similar results as in the Kalman case. 

Supporting files
--------------
**models/lgss_4parameters.py**
Defines the state space model that we use together with the expressions required to simulate from the model and estimate the gradient. 

**models/models_dists.py**
Subroutines for evaluating distributions, their derivatives and Hessians.

**models/models_helpers.py**
Subroutines for data generation and for importing data.

**para/pmh.py**
The main routine for the PMH algorithm and for estimating the Hessian using the quasi-Newton scheme.

**para/pmh_helpers.py**
Subroutines for exporting the data generated by the PMH algorithm.

**results/**
Data and plots from running the two RUNME-files.

**state/kalman.py**
The routines for Kalman filtering and smoothing to estimate the log-likelihood and gradients of the log-posterior.

**state/smc.py**
The routines for particle filtering and particle fixed-lag smoothing to estimate the log-likelihood and gradients of the log-posterior.
