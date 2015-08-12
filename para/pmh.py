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

import numpy       as     np
from   pmh_helpers import *
import pandas

##########################################################################
# Main class
##########################################################################

class stPMH(object):

    # Initalise some variables
    memoryLength      = None;
    empHessian        = None;

    ##########################################################################
    # Main sampling routine
    ##########################################################################

    def runSampler(self,sm,sys,thSys,PMHtype):

        #=====================================================================
        # Initalisation
        #=====================================================================

        # Set file prefix from model
        self.filePrefix = thSys.filePrefix;
        self.iter       = 0;
        self.PMHtype    = PMHtype;
        self.nPars      = thSys.nParInference;

        # Allocate vectors
        self.ll             = np.zeros((self.nIter,1))
        self.llp            = np.zeros((self.nIter,1))
        self.th             = np.zeros((self.nIter,self.nPars))
        self.tho            = np.zeros((self.nIter,self.nPars))
        self.thp            = np.zeros((self.nIter,self.nPars))
        self.aprob          = np.zeros((self.nIter,1))
        self.accept         = np.zeros((self.nIter,1))
        self.gradient       = np.zeros((self.nIter,self.nPars))
        self.gradientp      = np.zeros((self.nIter,self.nPars))
        self.hessian        = np.zeros((self.nIter,self.nPars,self.nPars))
        self.hessianp       = np.zeros((self.nIter,self.nPars,self.nPars))
        self.prior          = np.zeros((self.nIter,1))
        self.priorp         = np.zeros((self.nIter,1))
        self.J              = np.zeros((self.nIter,1))
        self.Jp             = np.zeros((self.nIter,1))
        self.proposalProb   = np.zeros((self.nIter,1))
        self.proposalProbP  = np.zeros((self.nIter,1))
        self.llDiff         = np.zeros((self.nIter,1))

        # Get the order of the PMH sampler
        if   ( PMHtype == "pPMH0" ):
            self.PMHtypeN        = 0;
        elif ( PMHtype == "pPMH1" ):
            self.PMHtypeN        = 1;
        elif ( PMHtype == "qPMH2" ):
            self.PMHtypeN        = 2;
            self.nHessianSamples = np.zeros((self.nIter,1))

        # Initialise the parameters in the proposal
        thSys.storeParameters(self.initPar,sys);

        # Run the initial filter/smoother
        self.estimateLikelihoodGradients(sm,thSys);
        self.acceptParameters(thSys);

        # Save the current parameters
        self.th[0,:]  = thSys.returnParameters();

        #=====================================================================
        # Main MCMC-loop
        #=====================================================================
        for kk in range(1,self.nIter):

            self.iter = kk;

            # Propose parameters
            self.sampleProposal();
            thSys.storeParameters( self.thp[kk,:], sys );

            # Calculate acceptance probability
            self.calculateAcceptanceProbability( sm, thSys );

            # Accept/reject step
            if ( np.random.random(1) < self.aprob[kk] ):
                self.acceptParameters( thSys );
            else:
                self.rejectParameters( thSys );

            # Write out progress report
            if np.remainder( kk, 100 ) == 0:
                progressPrint( self );

        progressPrint(self);

    ##########################################################################
    # Sample the proposal
    ##########################################################################
    def sampleProposal(self,):

        if ( self.PMHtype == "pPMH0" ):
            # Sample the preconditioned PMH0 proposal
            self.thp[self.iter,:] = self.th[self.iter-1,:] + np.random.multivariate_normal(np.zeros(self.nPars), self.stepSize**2 * self.invHessian );

        if( self.PMHtype == "pPMH1" ):
            # Sample the preconditioned PMH1 proposal
            self.thp[self.iter,:] = self.th[self.iter-1,:] + 0.5 * self.stepSize**2 * np.dot(self.invHessian,self.gradient[self.iter-1,:]) + np.random.multivariate_normal(np.zeros(self.nPars), self.stepSize**2 * self.invHessian );

        if ( self.PMHtype == "qPMH2" ):
            # Using PMH0 in initial phase and then quasi-Newton proposal
            if ( self.iter > self.memoryLength ):
                self.thp[self.iter,:] = self.th[self.iter-1-self.memoryLength,:] + 0.5 * self.stepSize**2 * np.dot( self.gradient[self.iter-1-self.memoryLength,:], self.hessian[self.iter-1-self.memoryLength,:,:] ) + np.random.multivariate_normal(np.zeros(self.nPars), self.stepSize**2 * self.hessian[self.iter-1-self.memoryLength,:,:] );
            else:
                self.thp[self.iter,:] = self.th[self.iter-1,:] + np.random.multivariate_normal( np.zeros(self.nPars), self.stepSize**2 * self.hessian[self.iter-1,:,:] );

    ##########################################################################
    # Calculate Acceptance Probability
    ##########################################################################
    def calculateAcceptanceProbability(self, sm,  thSys, ):

        # Run the smoother to get estimates of the log-likelihood and gradiets
        self.estimateLikelihoodGradients(sm,thSys);

        # Compute the part in the acceptance probability related to the non-symmetric proposal
        if ( self.PMHtype == "pPMH0" ):
            proposalP = 0;
            proposal0 = 0;

        if ( self.PMHtype == "pPMH1" ):
            proposalP = lognormpdf( self.thp[self.iter,:], self.th[self.iter-1,:]  + 0.5 * self.stepSize**2 * np.dot( self.invHessian,self.gradient[self.iter-1,:]),  self.stepSize**2 * self.invHessian  );
            proposal0 = lognormpdf( self.th[self.iter-1,:],self.thp[self.iter,:]   + 0.5 * self.stepSize**2 * np.dot( self.invHessian,self.gradientp[self.iter,:]) ,  self.stepSize**2 * self.invHessian  );

        if ( self.PMHtype == "qPMH2" ):

            if ( self.iter > self.memoryLength ):
                proposalP = lognormpdf( self.thp[self.iter,:],                     self.th[self.iter-1-self.memoryLength,:]  + 0.5 * self.stepSize**2 * np.dot( self.gradient[self.iter-1-self.memoryLength,:],  self.hessian[self.iter-1-self.memoryLength,:,:])  , self.stepSize**2 * self.hessian[self.iter-1-self.memoryLength,:,:]  );
                proposal0 = lognormpdf( self.th[self.iter-1-self.memoryLength,:],  self.thp[self.iter,:]                     + 0.5 * self.stepSize**2 * np.dot( self.gradientp[self.iter,:],                     self.hessianp[self.iter,:,:]) ,                     self.stepSize**2 * self.hessianp[self.iter,:,:] );
            else:
                # Initial phase, use pPMH0
                proposalP = lognormpdf( self.thp[self.iter,:],   self.th[self.iter-1,:]   , self.stepSize**2 * self.hessian[self.iter-1,:,:] );
                proposal0 = lognormpdf( self.th[self.iter-1,:],  self.thp[self.iter,:]    , self.stepSize**2 * self.hessianp[self.iter,:,:]  );

        # Compute the log-prior
        self.priorp[ self.iter ]    = thSys.prior();

        # Compute the acceptance probability
        self.aprob[ self.iter ] = self.flag * np.exp( self.llp[ self.iter, :] - self.ll[ self.iter-1, :] + proposal0 - proposalP + self.priorp[ self.iter, :] - self.prior[ self.iter-1, :] );

        # Store the proposal calculations
        self.proposalProb[ self.iter ]  = proposal0;
        self.proposalProbP[ self.iter ] = proposalP;
        self.llDiff[ self.iter ]        = self.llp[ self.iter, :] - self.ll[ self.iter-1, :];

    ##########################################################################
    # Run the SMC algorithm and get the required information
    ##########################################################################
    def estimateLikelihoodGradients(self,sm,thSys,):

        # Flag if the Hessian is PSD or not.
        self.flag  = 1.0

        # PMH0, only run the filter and extract the likelihood estimate
        if ( self.PMHtypeN == 0 ):
            sm.filter(thSys);

        # PMH1, only run the smoother and extract the likelihood estimate and gradient
        if ( self.PMHtypeN == 1 ):
            sm.smoother(thSys);
            self.gradientp[ self.iter,: ]   = sm.gradient;

        # PMH2, only run the smoother and extract the likelihood estimate and gradient
        if ( self.PMHtypeN == 2 ):
            sm.smoother(thSys);
            self.gradientp[ self.iter,: ]   = sm.gradient;

            # Note that this is the inverse Hessian
            self.hessianp [ self.iter,:,: ] = self.lbfgs_hessian_update( );

            # Extract the diagonal if needed and regularise if not PSD
            self.checkHessian();

        # Create output
        self.llp[ self.iter ]        = sm.ll;

        return None;

    ##########################################################################
    # Extract the diagonal if needed and regularise if not PSD
    ##########################################################################
    def checkHessian(self):

        # Pre-calculate posterior covariance estimate
        if ( ( self.iter >= self.nBurnIn ) & ( self.empHessian == None ) ) :
            self.empHessian = np.cov( self.th[range( self.nBurnIn - self.PSDmethodhybridSamps, self.nBurnIn ),].transpose() );

        # Check if it is PSD
        if ( ~isPSD( self.hessianp [ self.iter,:,: ] ) ):

            eigens = np.linalg.eig(self.hessianp [ self.iter,:,: ])[0];

            # Add a diagonal matrix proportional to the largest negative eigv during burnin
            if ( self.iter <= self.nBurnIn ):
                mineigv = np.min( np.linalg.eig( self.hessianp [ self.iter,:,: ] )[0] )
                self.hessianp [ self.iter,:,: ] = self.hessianp [ self.iter,:,: ] - 2.0 * mineigv * np.eye( self.nPars )
                print("Iteration: " + str(self.iter) + " has eigenvalues: " + str( eigens ) + " mirroring by adding " +  str( - 2.0 * mineigv ) );

            # Replace the Hessian with the posterior covariance matrix after burin
            if ( self.iter > self.nBurnIn ):
                self.hessianp [ self.iter,:,: ] = self.empHessian;
                print("Iteration: " + str(self.iter) + " has eigenvalues: " + str( eigens ) + " replaced Hessian with pre-computed estimated." );

    ##########################################################################
    # Quasi-Netwon proposal
    ##########################################################################
    def lbfgs_hessian_update(self):

        I  = np.eye(self.nPars, dtype=int);
        Hk = np.eye(self.nPars) / self.epsilon;

        # BFGS update for Hessian estimate
        if ( self.iter > self.memoryLength ):

            # Extract estimates of log-likelihood and gradients from the
            # last moves
            self.extractUniqueElements();

            if ( self.nHessianSamples[ self.iter ] > 2 ):
                # Extract the last unique parameters and their gradients
                idx = np.sort( np.unique(self.ll,return_index=True)[1] )[-2:];

                if ( np.max( self.iter - idx ) < self.memoryLength ):

                    # The last accepted step is inside of the memory length
                    skk = self.th[ idx[1] , : ]       - self.th[ idx[0], : ];
                    ykk = self.gradient[ idx[1] , : ] - self.gradient[ idx[0], : ];
                    foo = np.abs( np.dot( skk, ykk) / np.dot( ykk, ykk) );
                    Hk  = np.eye(self.nPars) * foo;

                # Add the contribution from the last memoryLength samples
                for ii in range( (self.thU).shape[0] ):

                    # Calculate difference in gradient (ykk) and theta (skk)
                    ykk = self.gradientU[ii,:] - self.gradientU[ii-1,:];
                    skk = self.thU[ii,:]       - self.thU[ii-1,:];

                    # Check if we have moved, otherwise to not add contribution to Hessian
                    if ( np.sum( skk ) != 0.0 ):

                        # Compute rho
                        rhok = 1.0 / ( np.dot( ykk, skk) );

                        # Update Hessian estimate
                        A1   = I - skk[:, np.newaxis] * ykk[np.newaxis, :] * rhok
                        A2   = I - ykk[:, np.newaxis] * skk[np.newaxis, :] * rhok
                        Hk   = np.dot(A1, np.dot(Hk, A2)) + ( rhok * skk[:, np.newaxis] * skk[np.newaxis, :] )

                # Return the negative Hessian estimate
                Hk = -Hk;

        return Hk;

    ##########################################################################
    # Helper to Quasi-Netwon proposal:
    # Extract the last n unique parameters and their gradients
    ##########################################################################

    def extractUniqueElements(self):

        # Find the unique elements
        idx            = np.sort( np.unique(self.ll[0:(self.iter-1)],return_index=True)[1] );

        # Extract the ones inside the memory length
        idx            = [ii for ii in idx if ii >= (self.iter - self.memoryLength) ]

        # Sort the indicies according to the log-likelihood
        idx2           = np.argsort( self.ll[idx], axis=0 )[:,0]

        # Get the new indicies
        idx3 = np.zeros( len(idx), dtype=int )
        for ii in idx2:
            idx3[ii] = idx[ii]

        # Extract and export the parameters and their gradients
        self.thU       = self.th[idx3,:];
        self.gradientU = self.gradient[idx3,:];

        # Save the number of indicies
        self.nHessianSamples[ self.iter ] = np.max((0,(self.thU).shape[0] - 1));

    ##########################################################################
    # Compute the IACT
    ##########################################################################

    def calcIACT( self, nSamples=None ):
        IACT = np.zeros( self.nPars );

        for ii in range( self.nPars ):
            if ( nSamples == None ):
                IACT[ii] = proto_IACT( self.th[self.nBurnIn:self.nIter,ii] )
            else:
                if ((  self.nIter-nSamples ) > 0 ):
                    IACT[ii] = proto_IACT( self.th[(self.nIter-nSamples):self.nIter,ii] )
                else:
                    raise NameError("More samples to compute IACT than iterations of the PMH algorithm.")

        return IACT

    ##########################################################################
    # Helper if parameters are accepted
    ##########################################################################
    def acceptParameters(self,thSys,):
        self.th[self.iter,:]        = self.thp[self.iter,:];
        self.tho[self.iter,:]       = thSys.returnParameters();
        self.ll[self.iter]          = self.llp[self.iter];
        self.gradient[self.iter,:]  = self.gradientp[self.iter,:];
        self.hessian[self.iter,:,:] = self.hessianp[self.iter,:];
        self.accept[self.iter]      = 1.0;
        self.prior[self.iter,:]     = self.priorp[self.iter,:];
        self.J[self.iter,:]         = self.Jp[self.iter,:];

    ##########################################################################
    # Helper if parameters are rejected
    ##########################################################################
    def rejectParameters(self,thSys,):
        self.th[self.iter,:]        = self.th[self.iter-1,:];
        self.tho[self.iter,:]       = self.tho[self.iter-1,:];
        self.ll[self.iter]          = self.ll[self.iter-1];
        self.prior[self.iter,:]     = self.prior[self.iter-1,:]
        self.gradient[self.iter,:]  = self.gradient[self.iter-1,:];
        self.hessian[self.iter,:,:] = self.hessian[self.iter-1,:,:];
        self.J[self.iter,:]         = self.J[self.iter-1,:];

    ##########################################################################
    # Helper: compile the results and write to file
    ##########################################################################
    def writeToFile(self,sm=None,fileOutName=None):

        # Set file name from parameter
        if ( ( self.fileOutName != None ) & (fileOutName == None) ):
            fileOutName = self.fileOutName;

        # Calculate the natural gradient
        ngrad = np.zeros((self.nIter,self.nPars));

        if ( self.PMHtype == "pPMH1" ):
            for kk in range(0,self.nIter):
                ngrad[kk,:] = np.dot( self.gradient[kk,:], self.invHessian );

        # Construct the columns labels
        columnlabels = [None]*(3*self.nPars+3);
        for ii in xrange(3*self.nPars+3):  columnlabels[ii] = ii;

        for ii in range(0,self.nPars):
            columnlabels[ii]               = "th" + str(ii);
            columnlabels[ii+self.nPars]    = "thp" + str(ii);
            columnlabels[ii+2*self.nPars]  = "ng" + str(ii);

        columnlabels[3*self.nPars]   = "acceptProb";
        columnlabels[3*self.nPars+1] = "loglikelihood";
        columnlabels[3*self.nPars+2] = "acceptflag";

        # Compile the results for output
        out = np.hstack((self.th,self.thp,ngrad,self.aprob,self.ll,self.accept));

        # Write out the results to file
        fileOut = pandas.DataFrame(out,columns=columnlabels);

        ensure_dir(fileOutName);
        fileOut.to_csv(fileOutName);

        print("writeToFile: wrote results to file: " + fileOutName)

##############################################################################
##############################################################################
# End of file
##############################################################################
##############################################################################
