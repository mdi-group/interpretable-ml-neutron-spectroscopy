import os
import numpy as np
import pickle
import time

start = 0
limit = 10000
date = 100
shape = (1679-1100, 1475-1100)
debug = False
index = 6
data_dir = '/work3/isis/scarfxxx/pcsmo_dimer/conv/'
label_dir = '/work3/isis/scarfxxx/pcsmo_dimer/label/'

det_mask = np.load('/work3/isis/scarfxxx/pcsmo/detector_mask.npy')

def build_dataframe(imgpath, labelpath, imgformat='tif', start=0, limit=10000, date_cut = 3000):
    '''
    Builds a pandas dataframe of paths to image files and assocaited labels, 
    which are stored 
    in the yaml file associated with that iamge.
    Args:
        imgpath: The directory where the files are (string)
        labelpath: The directory where the label files are (string)
        imgformat: The type of image files to look for
        start: the first index to start from
        limit: The upper limit on number of files to look at
        date_cut: oldest number of days old a file can be to be included, default 3000
    Returns:
        images, labels: a tuple of numpy arrays of the images and corresponding labels
    '''
    now = time.time()
    labels = []
    images = []
    files = [f for f in os.listdir(imgpath) if  
             os.path.isfile(os.path.join(imgpath, f)) and f.endswith(imgformat) and
             os.stat(imgpath + f).st_mtime > now - date_cut * 86400]
    files.sort(key=lambda x: os.path.getmtime(imgpath + x)) # sort by creation time
    files = files[start:limit]
    print('Found, {} image files'.format(len(files)))
    for i, file in enumerate(files[:limit]):
        if os.path.isfile(imgpath + file): 
            filename = imgpath + file       
            fileindex = filename.split('.')[0].split('/')[-1].split('_')[-1]
            labfile = labelpath + 'labels_' + fileindex + '.' + imgformat
            labels.append(np.load(labfile, encoding='bytes'))
            images.append(_build_spectrum(np.load(filename, encoding='bytes')))

    if len(files) > 0: 
        return np.array(images), np.array(labels)
    else:
        print('No files matching search criteria')
        return []

def _build_spectrum(simulated_image):

    for nei in range(len(simulated_image)):
        # Don't need to histogram as we've already done it in the resolution convolution
        if not simulated_image[nei]:
            if nei != 0:
                multispec = np.hstack((multispec, spectrum*0))
            continue
        ene_bins = [[s[b'i'], simulated_image[nei][b'en']] for s in simulated_image[nei][b'data']]
        q_pos = [s[b'q'] for s in simulated_image[nei][b'data']]
        nb = int(np.sqrt(len(simulated_image[nei][b'data'])))
    
        signal = np.array([e[0] for e in ene_bins])
        ne = np.shape(signal)[1]
        for x in range(ne):
            ene0 = np.array([e[x] for e in signal])
            s = np.array(ene0).reshape((nb, nb))
            if x == 0:
                spectrum = s
            else:
                spectrum = np.concatenate((spectrum, s))
        if nei == 0:
            multispec = spectrum
        else:
            multispec = np.hstack((multispec, spectrum))

    # Transpose image to look right for imshow, and mask pixels where exp data has no detectors
    multispec = multispec.T
    multispec[det_mask] = 0
    return multispec.reshape(len(simulated_image)*nb, ne*nb, 1)

imgs, labels = build_dataframe(data_dir, label_dir, imgformat='npy',
                     start=start, limit=limit, date_cut=date)
if len(imgs) > 0:
    np.save('simulated.npy', imgs)
    np.save('labels.npy', labels)
