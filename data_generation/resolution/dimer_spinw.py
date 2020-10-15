import numpy as np
import os
mcrpath = '/home/cm65/ml4n/lib/mcr2017b/v93/runtime/glnxa64:/home/cm65/ml4n/lib/mcr2017b/v93/bin/glnxa64:' \
          '/home/cm65/ml4n/lib/mcr2017b/v93/sys/os/glnxa64'
if 'LD_LIBRARY_PATH' not in os.environ:
    os.environ['LD_LIBRARY_PATH'] = mcrpath
else:
    os.environ['LD_LIBRARY_PATH'] += mcrpath
import sys
sys.path.append('/home/cm65/ml4n/lib')
import spinw
from matlab import double as md

MATLABENGINE = None

class dimer_spinw(object):
    """
    Class to encapsulate calculation of dispersion for Pr0.9(Ca,Sr)0.1Mn2O7 using SpinW (Matlab)
    """

    def __init__(self, spinw_path = os.path.join(os.path.expanduser('~'), 'src/spinw')):
        global MATLABENGINE
        if 'MATLABENGINE' not in globals() or MATLABENGINE is None:
            #MATLABENGINE = matlab.engine.start_matlab()
            spinw.initialize_runtime(['-nodisplay'])
            MATLABENGINE = spinw.initialize()
        self.m = MATLABENGINE
        self.m.addpath(self.m.genpath(spinw_path))
        self.obj = self.m.dimer_model(md([0]*6))

    def set_j(self, jvec=None):
        """
        Set the exchange parameters - assumed that parameters are in this order:
            [JFW, JA1, JA2, JFS, Jperp, D]
        The exchanges denoted are described in Johnstone et al., Phys. Rev. Lett. 109 237202 (2012)
        """
        if not jvec:
            raise RuntimeError('You must give an input J vector (array) which can have up to 6 elements and is '
                               'assumed to give the parameters in the following order: [JF1, JA, JF2, JF3, Jperp, D]')
        jvec = [j for j in jvec] + [0] * (6 - len(jvec)) # Sets undefined exchanges to zero
        self.m.matparser(self.obj, 'param', md(jvec), 'mat', ['JFW', 'JA1', 'JA2', 'JFS', 'Jperp', 'D(3,3)'], nargout=0)
        self.m.ass('swobj', self.obj, nargout=0)
        # Define twins
        self.m.ev('swobj.twin.rotc(:,:,2) = [0 1 0; 1 0 0; 0 0 0];', nargout=0)
        self.m.ev('swobj.twin.vol = [0.5 0.5];', nargout=0)

    def get_omega(self, q=None):
        """
        Calculates the dispersion omega(q) for a given momentum transfer q=[h,k,l].
        q can be a vector
        """
        if q is None:
            raise RuntimeError('You must specify a (or a set of) q vector(s).')
        ss = np.shape(q)
        if not (((len(ss) == 2) and (ss[0]==3 or ss[1]==3)) or ((len(ss) == 1) and ss[0]==3)):
            raise RuntimeError('Q vector is wrong shape - it must be a 3xN or Nx3 matrix or a 3-vector')
        qq = md(q)
        if ss[0] == 3:
            qq = self.m.transpose(md(q))
        self.m.ass('qq', qq, nargout=0)
        self.m.ev('spe = sw_neutron(swobj.spinwave(qq, "hermit", false, "formfact", 1, "fid", 0, "tid", 0));', nargout=0)
        #return self.m.evo('spe.omega', nargout=1), self.m.evo('spe.Sperp', nargout=1)
        # For twins, SpinW returns a cell array which is translated into Python as a list of Matlab arrays.
        # We concatenate the arrays within Matlab to return a single Matlab array instead
        return self.m.evo('[spe.omega{1}; spe.omega{2}]', nargout=1), self.m.evo('[spe.Sperp{1}; spe.Sperp{2}]', nargout=1)

    
if __name__ == '__main__':
    pcsmo = dimer_spinw()
    pcsmo.set_j([-8.43, 1.52, 1.52, -14.2, 0.92, 0.073])
    #print(pcsmo.get_omega([0.25, 0.45, 0.55]))
    for om, inten in zip(*pcsmo.get_omega([0.25, 0.45, 0.55])):
        print(om, inten)
