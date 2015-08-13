##############################################################################
##############################################################################
# Example code for
# quasi-Newton particle Metropolis-Hastings
# for a linear Gaussian state space model
#
# Please cite:
#
# J. Dahlin, F. Lindsten, T. B. Sch\"{o}n
# "Quasi-Newton particle Metropolis-Hastings"
# Proceedings of the 17th IFAC Symposium on System Identification,
# Beijing, China, October 2015.
#
# (c) 2015 Johan Dahlin
# johan.dahlin (at) liu.se
#
# Distributed under the MIT license.
#
##############################################################################
##############################################################################

import numpy            as np
import matplotlib.pylab as plt

from   state   import smc
from   para    import pmh
from   models  import lgss_4parameters


##############################################################################
# Arrange the data structures
##############################################################################
sm               = smc.smcSampler();
ppmh0            = pmh.stPMH();
ppmh1            = pmh.stPMH();
qpmh2            = pmh.stPMH();


##############################################################################
# Setup the system
##############################################################################
sys               = lgss_4parameters.ssm()
sys.par           = np.zeros((sys.nPar,1))
sys.par[0]        = 0.20;
sys.par[1]        = 0.80;
sys.par[2]        = 1.00;
sys.par[3]        = 0.10;
sys.T             = 250;
sys.xo            = 0.0;


##############################################################################
# Load data
##############################################################################
sys.generateData(fileName="data/lgssT250_smallR.csv",order="xy");


##############################################################################
# Setup the parameters
##############################################################################
th               = lgss_4parameters.ssm()
th.version       = "standard"
th.nParInference = 3;
th.copyData(sys);


##############################################################################
# Setup the SMC algorithm
##############################################################################

# Use the particle filter to estimate the log-likelihood
sm.filter          = sm.faPF;

# Use the fixed-lag smoother to estimate the gradient
sm.smoother        = sm.flPS;

# Use nPart number of particles
sm.nPart           = 50;

# Use fixedLag as Delta in the fixed-lag smoother
sm.fixedLag        = 12;

# Estimate the initial state
sm.genInitialState = True;

# Set seed to reproduce the results
np.random.seed( 87655678 );


##############################################################################
# Setup the PMH algorithm
##############################################################################

# Set number of total PMH iterations
ppmh0.nIter          = 5000;
ppmh1.nIter          = 5000;
qpmh2.nIter          = 5000;

# Set the length of the burn-in
ppmh0.nBurnIn        = 1000;
ppmh1.nBurnIn        = 1000;
qpmh2.nBurnIn        = 1000;

# Set initial parameters
ppmh0.initPar        = th.returnParameters();
ppmh1.initPar        = th.returnParameters();
qpmh2.initPar        = th.returnParameters();

# Set the pre-conditioning matrix (estimated from an initial run)
ppmh0.invHessian     = np.diag( (3e-02, 1e-03, 2e-03 ) )
ppmh1.invHessian     = np.diag( (3e-02, 1e-03, 2e-03 ) )


########################################################################
# Preconditioned PMH0 sampler
########################################################################

# Set the step size using the rule of thumb
ppmh0.stepSize       = 2.562 / np.sqrt(th.nParInference);

# Run the sampler
ppmh0.runSampler(sm, sys, th, "pPMH0");

# Write the output to file
ppmh0.fileOutName = 'results/lgss-fapf-ppmh0.csv'
ppmh0.writeToFile();

# Compute IACT
iactPPMH0 = ppmh0.calcIACT();


########################################################################
# Preconditioned PMH1 sampler
########################################################################

# Set the step size using the rule of thumb
ppmh1.stepSize       = 1.125 * np.sqrt( th.nParInference**(-1/3) );

# Run the sampler
ppmh1.runSampler(sm, sys, th, "pPMH1");

# Write the output to file
ppmh1.fileOutName = 'results/lgss-fapf-ppmh1.csv'
ppmh1.writeToFile();

# Compute IACT
iactPPMH1 = ppmh1.calcIACT();


########################################################################
# Quasi-Newton PMH2 sampler with hybrid correction
########################################################################

# Step size the sampler \epsilon_2
qpmh2.stepSize             = 1.0;

# Set the initial Hessian
qpmh2.epsilon              = 1000;

# Set the memory length of the quasi-Newton proposal
qpmh2.memoryLength         = 100;

# Use the hybrid form of the sampler (replacing a non-PSD Hessian with the
# estimated posterior covariance using the last iterations of the burn-in)
qpmh2.PSDmethodhybridSamps = 500;

# Run the sampler
qpmh2.runSampler(sm, sys, th, "qPMH2");

# Write the output to file
qpmh2.fileOutName = 'results/lgss-fapf-qpmh2.csv'
qpmh2.writeToFile();

# Compute IACT
iactQPMH2 = qpmh2.calcIACT();


########################################################################
# Plot the trace plots of the Markov chains
########################################################################

plt.figure(1);
plt.subplot(3,1,1); plt.plot(ppmh0.th); plt.title('Trace plot for pPMH0');
plt.xlabel('iteration'); plt.ylabel('trace'); plt.axis((ppmh0.nBurnIn,ppmh0.nIter,-1.0,1.5));

plt.subplot(3,1,2); plt.plot(ppmh1.th); plt.title('Trace plot for pPMH1');
plt.xlabel('iteration'); plt.ylabel('trace'); plt.axis((ppmh0.nBurnIn,ppmh0.nIter,-1.0,1.5));

plt.subplot(3,1,3); plt.plot(qpmh2.th); plt.title('Trace plot for qPMH2');
plt.xlabel('iteration'); plt.ylabel('trace'); plt.axis((ppmh0.nBurnIn,ppmh0.nIter,-1.0,1.5));

plt.legend(('mu','phi','sigmav'))


########################################################################
# Plot the posterior estimates from the PMH algorithms
########################################################################

plt.figure(2);
plt.subplot(3,3,1);
n, bins, patches = plt.hist(ppmh0.th[ppmh0.nBurnIn:ppmh0.nIter,0],np.floor(np.sqrt(ppmh0.nIter-ppmh0.nBurnIn)),normed=1,histtype='stepfilled');
plt.setp(patches, 'facecolor', 'b', 'alpha', 0.75); plt.xlabel('mu'); plt.ylabel('posterior estimate'); plt.axis((-0.8,0.8,0,4));

plt.subplot(3,3,2);
n, bins, patches = plt.hist(ppmh0.th[ppmh0.nBurnIn:ppmh0.nIter,1],np.floor(np.sqrt(ppmh0.nIter-ppmh0.nBurnIn)),normed=1,histtype='stepfilled');
plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75); plt.xlabel('phi'); plt.ylabel('posterior estimate'); plt.axis((0.7,1.0,0,20));

plt.subplot(3,3,3);
n, bins, patches = plt.hist(ppmh0.th[ppmh0.nBurnIn:ppmh0.nIter,2],np.floor(np.sqrt(ppmh0.nIter-ppmh0.nBurnIn)),normed=1,histtype='stepfilled');
plt.setp(patches, 'facecolor', 'r', 'alpha', 0.75); plt.xlabel('sigmav'); plt.ylabel('posterior estimate'); plt.axis((0.85,1.3,0,14));

plt.subplot(3,3,4);
n, bins, patches = plt.hist(ppmh1.th[ppmh1.nBurnIn:ppmh1.nIter,0],np.floor(np.sqrt(ppmh1.nIter-ppmh1.nBurnIn)),normed=1,histtype='stepfilled');
plt.setp(patches, 'facecolor', 'b', 'alpha', 0.75); plt.xlabel('mu'); plt.ylabel('posterior estimate'); plt.axis((-0.8,0.8,0,4));

plt.subplot(3,3,5);
n, bins, patches = plt.hist(ppmh1.th[ppmh1.nBurnIn:ppmh1.nIter,1],np.floor(np.sqrt(ppmh1.nIter-ppmh1.nBurnIn)),normed=1,histtype='stepfilled');
plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75); plt.xlabel('phi'); plt.ylabel('posterior estimate'); plt.axis((0.7,1.0,0,20));

plt.subplot(3,3,6);
n, bins, patches = plt.hist(ppmh1.th[ppmh1.nBurnIn:ppmh1.nIter,2],np.floor(np.sqrt(ppmh1.nIter-ppmh1.nBurnIn)),normed=1,histtype='stepfilled');
plt.setp(patches, 'facecolor', 'r', 'alpha', 0.75); plt.xlabel('sigmav'); plt.ylabel('posterior estimate'); plt.axis((0.85,1.3,0,14));

plt.subplot(3,3,7);
n, bins, patches = plt.hist(qpmh2.th[qpmh2.nBurnIn:qpmh2.nIter,0],np.floor(np.sqrt(qpmh2.nIter-qpmh2.nBurnIn)),normed=1,histtype='stepfilled');
plt.setp(patches, 'facecolor', 'b', 'alpha', 0.75); plt.xlabel('mu'); plt.ylabel('posterior estimate'); plt.axis((-0.8,0.8,0,4));

plt.subplot(3,3,8);
n, bins, patches = plt.hist(qpmh2.th[qpmh2.nBurnIn:qpmh2.nIter,1],np.floor(np.sqrt(qpmh2.nIter-qpmh2.nBurnIn)),normed=1,histtype='stepfilled');
plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75); plt.xlabel('phi'); plt.ylabel('posterior estimate'); plt.axis((0.7,1.0,0,20));

plt.subplot(3,3,9);
n, bins, patches = plt.hist(qpmh2.th[qpmh2.nBurnIn:qpmh2.nIter,2],np.floor(np.sqrt(qpmh2.nIter-qpmh2.nBurnIn)),normed=1,histtype='stepfilled');
plt.setp(patches, 'facecolor', 'r', 'alpha', 0.75); plt.xlabel('sigmav'); plt.ylabel('posterior estimate'); plt.axis((0.85,1.3,0,14));


########################################################################
# Compare the IACT values
########################################################################

print("IACT for pPMH0: " + str( np.round(iactPPMH0,0)) + ".");
print("IACT for pPMH1: " + str( np.round(iactPPMH1,0)) + ".");
print("IACT for qPMH2: " + str( np.round(iactQPMH2,0)) + ".");

#IACT for pPMH0: [ 12.  12.  11.].
#IACT for pPMH1: [ 6.  6.  8.].
#IACT for qPMH2: [ 4.  3.  4.].

########################################################################
# End of file
########################################################################
