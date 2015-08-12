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

import numpy as np
import os

##############################################################################
# Print small progress reports
##############################################################################
def progressPrint(pmh):
    print("################################################################################################ ");
    print(" Iteration: " + str(pmh.iter+1) + " of : " + str(pmh.nIter) + " completed.")
    print("");
    print(" Current state of the Markov chain:               ")
    print(["%.4f" % v for v in pmh.th[pmh.iter,:]])
    print("");
    print(" Proposed next state of the Markov chain:         ")
    print(["%.4f" % v for v in pmh.thp[pmh.iter,:]])
    print("");
    print(" Current posterior mean estimate (untransformed): ")
    print(["%.4f" % v for v in np.mean(pmh.tho[range(pmh.iter),:], axis=0)])
    print("");
    print(" Current acceptance rate:                         ")
    print("%.4f" % np.mean(pmh.accept[range(pmh.iter)]) )
    if ( ( pmh.PMHtype == "qPMH2" ) & ( pmh.iter > pmh.memoryLength ) ):
        print("");
        print(" Mean no. samples for Hessian estimate:           ")
        print("%.4f" % np.mean(pmh.nHessianSamples[range(pmh.memoryLength,pmh.iter)]) )
#    print("");
#    print(" Current IACT: ")
#    print(["%.2f" % v for v in IACT(pmh.th[0:pmh.iter,:]) ])
    print("################################################################################################ ");

##############################################################################
# Check if dirs for outputs exists, otherwise create them
##############################################################################
def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

##############################################################################
# Calculate the Integrated Autocorrlation Time
##############################################################################
def proto_IACT( x ):

    # estimate the acf coefficient
    # code from http://stackoverflow.com/questions/14297012/estimate-autocorrelation-using-python
    n = len(x)
    nmax = int(np.floor(n/10.0));
    data = np.asarray(x)
    mean = np.mean(data)
    c0 = np.sum((data - mean) ** 2) / float(n)

    def r(h):
        acf_lag = ((data[:n - h] - mean) * (data[h:] - mean)).sum() / float(n) / c0
        return round(acf_lag, 3)
    x = np.arange(n) # Avoiding lag 0 calculation
    acf_coeffs = map(r, x)

    try:
        cutoff = np.where( np.abs( acf_coeffs[0:int(nmax)] ) < 2.0 / np.sqrt(n) )[0][0];
    except:
        cutoff = n;

    # estimate the maximum number of acf coefficients to use to calculate IACT
    tmp  = int( min(cutoff,nmax) );

    # estimate the IACT
    iact = 1.0 + 2.0 * np.sum( acf_coeffs[0:tmp] );

    return iact;

##############################################################################
# Calculate the log-pdf of a univariate Gaussian
##############################################################################
def loguninormpdf(x,mu,sigma):
    return -0.5 * np.log( 2.0 * np.pi * sigma**2) - 0.5 * (x-mu)**2 * sigma**(-2);

##############################################################################
# Calculate the log-pdf of a multivariate Gaussian with mean vector mu and covariance matrix S
##############################################################################
def lognormpdf(x,mu,S):
    nx = len(S)
    norm_coeff = nx * np.log( 2.0 * np.pi ) + np.linalg.slogdet(S)[1]
    err = x-mu

    numerator = np.dot( np.dot(err,np.linalg.pinv(S)),err.transpose())
    return -0.5*(norm_coeff+numerator)

##############################################################################
# Check if a matrix is positive semi-definite but checking for negative eigenvalues
##############################################################################
def isPSD(x):
    return np.all(np.linalg.eigvals(x) > 0)

##############################################################################
##############################################################################
# End of file
##############################################################################
##############################################################################