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

import numpy                 as     np
import scipy                 as     sp

##############################################################################
# Gamma pdf, gradient and Hessian of log-pdf
##############################################################################

def gammaPDF(x, a, b):
    # Shape and rate parametrisation
    return b**a / sp.special.gamma(a) * x**(a-1.0) * np.exp(-b*x)

def gammaLogPDF(x, a, b):
    # Shape and rate parametrisation
    return a * np.log(b) + (a-1.0) * np.log(x) - b * x - sp.special.gammaln(a)

def gammaLogPDFgradient (x, a, b):
    # Shape and rate parametrisation
    return ( a - 1.0 ) / x - b;

def gammaLogPDFhessian (x, a, b):
    # Shape and rate parametrisation
    return - ( a - 1.0 ) / ( x**2 );

##############################################################################
# Beta pdf, gradient and Hessian of log-pdf
##############################################################################

def betaPDF(x, a, b):
    return sp.special.gamma(a+b) / ( sp.special.gamma(a) + sp.special.gamma(b) ) * x**(a-1.0) * (1.0-x)**(b-1.0);

def betaLogPDF(x, a, b):
    return sp.special.gammaln(a+b) - np.log( sp.special.gamma(a) + sp.special.gamma(b) ) + (a-1.0)*np.log(x) * (b-1.0) * np.log(1.0-x);

def betaLogPDFgradient (x, a, b):
    return ( a - 1.0 ) / x + ( 1.0 - b ) / ( 1.0 - x );

def betaLogPDFhessian (x, a, b):
    return - ( a - 1.0 ) / ( x**2 ) + ( 1.0 - b ) / ( ( 1.0 - x )**2 );

##############################################################################
# Normal pdf, gradient and Hessian of log-pdf
##############################################################################

def normalPDF(x, a, b):
    return 1.0 / np.sqrt( 2 * np.pi * b**2 ) * np.exp( -0.5/(b**2) * (x-a)**2 );

def normalLogPDF(x, a, b):
    return -0.5 * np.log( 2 * np.pi * b**2 ) -0.5/(b**2) * (x-a)**2;

def normalLogPDFgradient (x, a, b):
    return (a - x)/b**2;

def normalLogPDFhessian (x, a, b):
    return -1.0/b**2

##############################################################################
# Logit, inv-logit and derivatives
##############################################################################

def sigmoid(x):
  return 1.0 / ( 1.0 + np.exp(-x) )

def invSigmoid(x):
  return np.log( x ) - np.log( 1.0 - x );

def sigmoidGradient(x):
  return sigmoid(x) * ( 1.0 - sigmoid(x) );

def invSigmoidGradient(x):
  return - 1.0 / ( ( x - 1.0 ) * x );


##############################################################################
##############################################################################
# End of file
##############################################################################
##############################################################################
