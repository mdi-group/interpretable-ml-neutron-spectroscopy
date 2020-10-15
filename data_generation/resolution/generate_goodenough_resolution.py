###############VALUES TO SET
prefix = '/work3/isis/scarfxxx/'
data_path = '/work3/isis/scarfxxx/pcsmo_goodenough/'
J1 = 11.39   # Gap 
J2 = -1.50   # Gap
J3 = 1.35    #
J4 = -1.50   # Lower energy dispersion - lower energy data required <= 70
J5 = -0.88   # Lower energy dispersion - lower energy data required <= 70
J6 = 0.074
ei = 70
run_multiEi = True # Whether to run for just the specified Ei or for all Ei's
filename = 'resolution_spec_'
labelfile = 'labels_'
run_spinw = True # Whether to run the SpinW calculation or just the resolution convolution
# The numbers here govern the grid density of the output (resolution convolved) and SpinW grids.
# Default is both is 40x40 but they can be independent.
# The grid always extends from (h,k)=(-1,-1) to (+1,+1).
nb = 40  # Number of output grid points (output grid will be nb x nb)
nc = 40  # Number of SpinW calculated points (spinw grid will be nc x nc)
nl = 1   # Number of L points to calculate (in principle the dispersion is flat along L)
ne = 10  # Number of energy bins
# The run index for the generated datasets.
# If it is 1-element, runs with fitted J pars, else generate random J's
# This is overwritten if a commandline argument is supplied (being the index)
run_index = [0]
#################################

import sys
libdir = prefix + 'lib/'
sys.path.append(libdir)
from resolution_function import readMat, covariance
import numpy as np
import time

import os
mcrpath = '{}/mcr2017b/v93/runtime/glnxa64:{}/mcr2017b/v93/bin/glnxa64:{}/mcr2017b/v93/sys/os/glnxa64'.format(libdir, libdir, libdir)
if 'LD_LIBRARY_PATH' not in os.environ:
    os.environ['LD_LIBRARY_PATH'] = mcrpath
else:
    os.environ['LD_LIBRARY_PATH'] += mcrpath
from goodenough_spinw import goodenough_spinw
from matlab import double as md

def do_convolution(output_h, output_k, output_e, data, resolution_table, double_e=False):
    dE = np.mean(np.diff(output_e))
    if dE > covariance(resolution_table, [0, 0, 0])[0,0]:
        double_e = True
    e1 = np.linspace(np.min(output_e), np.max(output_e), len(output_e)+1)
    if double_e:
        # If the e-grid density is larger than the resolution, we double the density of the E-grid 
        dE1 = np.mean(np.diff(e1))
        output_e = np.concatenate([[e1[ie]+dE1/3, e1[ie]+dE1*2/3] for ie in range(len(output_e))])
    else:
        # Recast the output e-grid to be midpoints of bin ranges rather than the bin edges.
        output_e = np.array([(e1[ie]+e1[ie+1])/2 for ie in range(len(output_e))])
    hh, kk, ee = np.meshgrid(output_h, output_k, output_e)
    hh = hh.flatten()
    kk = kk.flatten()
    ee = ee.flatten()
    emin, emax = (np.min(output_e), np.max(output_e))
    nb, ne = (len(output_h), len(output_e))
    qe_grid = np.zeros((nb, nb, ne))
    for dat in data:
        for ie, en in enumerate(dat['en']):
            Qh = dat['q'][0] * 0.5
            Qk = dat['q'][1] * 0.5
            resmat = covariance(resolution_table, [en, Qh, Qk])
            # The covariance function gives a covariance matrix in this basis: [en, Qh, Qk]
            # The following code assumes it is is in this basis: [Qh, Qk, en] (as the original in the text)
            # So we need to permute both rows and columns.
            # https://stackoverflow.com/questions/20265229/rearrange-columns-of-numpy-2d-array
            resmat = resmat[:,[1,2,0]][[1,2,0]]
            # modes larger than ei are not kinematically possible, so the covariance matrix will be zero
            if (en < emin):# or (en > emax):
                continue
            # If we somehow cannot interpolate a resolution matrix at the Q point, ignore it.
            if (resmat == 0).all():
                continue
            # calc 3D gaussian centred at [h,k,omega] caclulated.
            pd, pv = np.linalg.eig(resmat)
            # We use the eigenvectors of the covariance matrix to transform the desired coordinates into
            # a basis where they are equivalent to the coordinates of a standard 3D Gaussian and then
            # calculate the intensities from this. See:
            # https://janakiev.com/blog/covariance-matrix/
            xx = np.matrix([hh - Qh, kk - Qk, ee - en]).T
            # We actually want the inverse of the following transformation matrix.
            #trans_mat = ((pv).dot(np.diag(np.sqrt(pd)))).T
            #x2 = xx.dot( np.linalg.inv( trans_mat ) )
            # But computing the inverse is expensive, so we do the matrix algebra slightly differently to get the inverse.
            # This is because the eigenvector matrix pv is orthogonal so its transpose is its inverse.
            x2 = xx.dot( np.diag(1./np.sqrt(pd)).dot(pv.T).T )
            gauss3d = np.exp(-(np.array(x2[:,0])**2)/2 - (np.array(x2[:,1])**2)/2 - (np.array(x2[:,2])**2)/2) / (2*np.pi)
            qe_grid = qe_grid + (np.reshape(gauss3d, (nb, nb, ne)) * dat['i'][ie])
            
    grid_data = []
    if double_e:
        for ih, h in enumerate(output_h):
            for ik, k in enumerate(output_k):
                intensities_array = [qe_grid[ih, ik, ie]+qe_grid[ih, ik, ie+1] for ie in range(0, ne-1, 2)]
                grid_data.append({'q':[h, k, 0], 'i':intensities_array})
    else:
        for ih, h in enumerate(output_h):
            for ik, k in enumerate(output_k):
                intensities_array = qe_grid[ih, ik, :]   # intensities vs energy
                grid_data.append({'q':[h, k, 0], 'i':intensities_array})
    return grid_data


# The following (h,k) values are in units of the high-temperature structural unit cell
# used by the PRL paper. The SpinW calculations uses a magnetic unit cell which is 2x2x1 structural units
# Hence the limits of the SpinW calculations go from -2 to +2 (which is -1 to +1 in the HT cell)
output_h = np.linspace(-1, 1, nb)
output_k = np.linspace(-1, 1, nb)
output_e = np.linspace(2, 50, ne)
emin = np.min(output_e)
emax = np.max(output_e)
spinw_h = np.linspace(-2, 2, nc)
spinw_k = np.linspace(-2, 2, nc)
spinw_l = np.linspace(-1, 1, nl)

if run_multiEi:
    restabs = {}
    for ei in [25, 35, 50, 70, 100, 140]:
        restabs[ei] = readMat('{}/PCSMO_ei{}_resolution.txt'.format(data_path, ei))
else:
    resolution_table = readMat('{}/PCSMO_ei{}_resolution.txt'.format(data_path, ei))

if len(sys.argv) > 1:
    i0 = int(sys.argv[1])
    #run_index = range(i0*10, i0*10+10)
    #run_index = range(i0*60+2440, i0*60+2500)
    remind = np.load('remainder.npy')
    #run_index = remind[(i0*20):(i0*20+20)]
    run_index = [remind[i0]]
print(run_index)

t0 = time.time()
for ind in run_index:
    fname = filename + str(ind)
    lname = labelfile + str(ind)
    if len(run_index) > 0:
        J1 = float(np.random.uniform(-20, 0))
        J2 = float(np.random.uniform(0, 3))
        J3 = float(np.random.uniform(-3, 3))
        J4 = float(np.random.uniform(-3, 3))
        J5 = float(np.random.uniform(0, 3))
        J6 = float(np.random.uniform(0, 0.2))
        print("Running index {} with Js=[{}, {}, {}, {}, {}, {}] at t={:5}".format(ind, J1, J2, J3, J4, J5, J6, time.time()-t0))
    Js = [J1, J2, J3, J4, J5, J6]
    if run_spinw:
         pcsmo = goodenough_spinw()
         pcsmo.set_j([J1, J2, J3, J4, J5, J6])
         data = []
         for kk in range(nl):
             for i in range(nc):
                 for j in range(nc):
                     h = spinw_h[i]
                     k = spinw_k[j]
                     l = 0 if (nl == 1) else spinw_l[kk]
                     q_pos = [h, k, l]
                     oms, intens = pcsmo.get_omega(q_pos)
                     for omq, intq in zip(np.array(oms).T, np.array(intens).T):
                         omegas = []
                         intensities = []
                         for om, intens in zip(omq, intq):
                             omegas.append(om.real)
                             intensities.append(intens.real)
                         #data.append([q_pos, np.histogram(omegas, bins=10, range=(2, 50), weights=intensities)])
                         data.append({'q':q_pos, 'en':omegas, 'i':intensities})
    else:
        unconvfile = "unconv/{}_unconvolted.npy".format(fname) if len(run_index) > 0 else "resolution_spec_unconvolted.npy"
        data = np.load('{}/{}'.format(data_path, unconvfile))

    
    eis = [25, 35, 50, 70, 100, 140] if run_multiEi else [ei]
    conv_data = []
    for ei in eis:
        if run_multiEi:
            output_e = np.linspace(ei*0.1, ei*0.7, ne)
            resolution_table = restabs[ei]
        #if ei == 35:
        #    conv_data.append([])
        #else:
        grid_data = do_convolution(output_h, output_k, output_e, data, resolution_table)
        conv_data.append({'ei': ei, 'en':output_e, 'data':grid_data})

    if len(run_index) > 0:
        if run_spinw:
            np.save(data_path + '/unconv/' + fname + "_unconvolted.npy", data)
            #print('JJJJJ', Js)
            np.save(data_path + '/label/' + lname + ".npy", Js)
        np.save(data_path + '/conv/' + fname + ".npy", conv_data)
    else:
        if run_spinw:
            np.save("{}/resolution_spec_unconvolted.npy".format(data_path), data)
        np.save("{}/resolution_spec.npy".format(data_path), conv_data)
