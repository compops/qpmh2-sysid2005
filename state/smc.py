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
import scipy.weave           as     weave

##############################################################################
# Main class
##############################################################################

class smcSampler(object):

    ##########################################################################
    # Particle filtering: wrappers for special cases
    ##########################################################################

    def bPF(self,sys):
        self.filePrefix               = sys.filePrefix;
        self.resamplingInternal       = 1;
        self.filterTypeInternal       = "bootstrap"
        self.condFilterInternal       = 0;
        self.ancestorSamplingInternal = 0;
        self.filterType               = "bPF";
        self.pf(sys);

    # Fully adapted particle filter
    def faPF(self,sys):
        self.filePrefix               = sys.filePrefix;
        self.resamplingInternal       = 1;
        self.filterTypeInternal       = "fullyadapted";
        self.condFilterInternal       = 0;
        self.ancestorSamplingInternal = 0;
        self.filterType               = "faPF";
        self.pf(sys);

    ##########################################################################
    # Particle filtering: main routine
    ##########################################################################

    def pf(self,sys):

        # Initalise variables
        a   = np.zeros((self.nPart,sys.T));
        p   = np.zeros((self.nPart,sys.T));
        pt  = np.zeros((self.nPart,sys.T));
        v   = np.zeros((self.nPart,sys.T));
        w   = np.zeros((self.nPart,sys.T));
        xh  = np.zeros((sys.T,1));
        ll  = np.zeros(sys.T);

        # Save T
        self.T = sys.T;

        # Generate the initial particles
        p[:,0] = sys.generateInitialState( self.nPart );

        #=====================================================================
        # Run main loop
        #=====================================================================
        for tt in range(0, sys.T):

            if tt != 0:
                #=============================================================
                # Resample particles
                #=============================================================

                # Systematic resampling
                nIdx     = self.resampleSystematic(w[:,tt-1]);
                nIdx     = np.transpose(nIdx.astype(int));
                pt[:,tt] = p[nIdx,tt-1];
                a[:,tt]  = nIdx;

                #=============================================================
                # Propagate particles
                #=============================================================
                if ( self.filterTypeInternal == "bootstrap" ):
                    p[:,tt] = sys.generateState   ( pt[:,tt], tt-1);
                elif ( ( self.filterTypeInternal == "fullyadapted" ) & (tt != (sys.T-1)) ):
                    p[:,tt] = sys.generateStateFA ( pt[:,tt], tt-1);

            #=================================================================
            # Weight particles
            #=================================================================
            if ( self.filterTypeInternal == "bootstrap" ):
                w[:,tt] = sys.evaluateObservation   ( p[:,tt], tt);
            elif ( ( self.filterTypeInternal == "fullyadapted" ) & (tt != (sys.T-1)) ):
                w[:,tt] = sys.evaluateObservationFA ( p[:,tt], tt);

            # Rescale log-weights and recover weights
            wmax    = np.max( w[:,tt] );
            w[:,tt] = np.exp( w[:,tt] - wmax );

            # Estimate log-likelihood
            ll[tt]   = wmax + np.log(np.sum(w[:,tt])) - np.log(self.nPart);
            w[:,tt] /= np.sum(w[:,tt]);

            # Calculate the normalised filter weights (1/N) as it is a FAPF
            if ( ( self.filterTypeInternal == "fullyadapted" ) & (tt != (sys.T-1)) ):
                v[:,tt] = w[:,tt];
                w[:,tt] = np.ones(self.nPart) / self.nPart;

            # Estimate the filtered state
            xh[tt]  = np.sum( w[:,tt] * p[:,tt] );

        #=====================================================================
        # Create output
        #=====================================================================
        self.xhatf = xh;
        self.ll    = np.sum( ll );
        self.llt   = ll;
        self.w     = w;
        self.v     = v;
        self.a     = a;
        self.p     = p;
        self.pt    = pt;

    ##########################################################################
    # Particle smoothing: fixed-lag smoother
    ##########################################################################

    def flPS(self,sys):

        #=====================================================================
        # Initalisation
        #=====================================================================

        # Check algorithm settings and set to default if needed
        self.T = sys.T;
        self.smootherType = "fl"

        # Run initial filter
        self.filter(sys);

        # Initalise variables
        xs    = np.zeros((sys.T,1));
        g1    = np.zeros((sys.nParInference,sys.T));

        #=====================================================================
        # Main loop
        #=====================================================================

        for tt in range(0, sys.T-1):
            at  = np.arange(0,self.nPart)
            kk  = np.min( (tt+self.fixedLag, sys.T-1) )

            # Reconstruct particle trajectory
            for ii in range(kk,tt,-1):
                att = at.astype(int);
                at  = at.astype(int);
                at  = self.a[at,ii];
                at  = at.astype(int);

            # Estimate state
            xs[tt] = np.sum( self.p[at,tt] * self.w[:, kk] );

            # Estimate the contribution to the gradient of the log-likelihood at time tt
            sa = sys.Dparm  ( self.p[att,tt+1], self.p[at,tt], np.zeros(self.nPart), at, tt);

            for nn in range(0,sys.nParInference):
                g1[nn,tt]       = np.sum( sa[:,nn] * self.w[:,kk] );

        # Estimate the gradient of the log-likelihood
        self.gradient = np.nansum(g1,axis=1);

        # Add the gradient of the log-prior
        for nn in range(0,sys.nParInference):
            self.gradient[nn]     = sys.dprior1(nn) + self.gradient[nn];

        # Save the smoothed state estimate
        self.xhats = xs;

    ##############################################################################
    # Systematic resampling
    ##############################################################################

    def resampleSystematic( self, w, N=0 ):
        code = \
        """ py::list ret;
    	int jj = 0;
            for(int kk = 0; kk < N; kk++)
            {
                double uu  = ( u + kk ) / N;

                while( ww(jj) < uu && jj < H - 1)
                {
                    jj++;
                }
                ret.append(jj);
            }
    	return_val = ret;
        """
        H = len(w);
        if N==0:
            N = H;

        u   = float( np.random.uniform() );
        ww  = ( np.cumsum(w) / np.sum(w) ).astype(float);
        idx = weave.inline(code,['u','H','ww','N'], type_converters=weave.converters.blitz )
        return np.array( idx ).astype(int);

##############################################################################
##############################################################################
# End of file
##############################################################################
##############################################################################
