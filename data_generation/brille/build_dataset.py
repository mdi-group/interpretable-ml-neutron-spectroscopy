import os
import numpy as np
import scipy.io
import time

start = 0
limit = 100000
date = 100
#shape = (1679-1100, 1475-1100)
debug = False
index = 6
#prefix = os.path.dirname(os.path.realpath(__file__))
prefix = os.getcwd()
data_dir = prefix + '/conv/'
label_dir = prefix + '/label/'

det_mask = scipy.io.loadmat('nanmask.mat')
det_mask = det_mask['nan_mask']

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
    marginal = 0
    for i, file in enumerate(files[:limit]):
        if os.path.isfile(imgpath + file): 
            filename = imgpath + file       
            mdic = scipy.io.loadmat(filename)
            mm = mdic['mat']
            mm[np.where(det_mask)] = 0
            if i == 0: sz = float(np.prod(np.shape(mm)))
            nanlev = np.sum(np.isnan(mm)) / sz
            if nanlev > 0: #0.01:
                continue
            #elif nanlev > 0:
            #    mm[np.where(np.isnan(mm))] = 0
            #    marginal += 1
            images.append(np.flipud(mm) * 4.)
            labels.append(np.squeeze(mdic['jvec']))

    if len(files) > 0: 
        print('{} good files; {} marginal'.format(np.shape(images)[0], marginal))
        return np.array(images), np.array(labels)
    else:
        print('No files matching search criteria')
        return []

imgs, labels = build_dataframe(data_dir, label_dir, imgformat='mat',
                     start=start, limit=limit, date_cut=date)
if len(imgs) > 0:
    np.save('simulated.npy', imgs)
    np.save('labels.npy', labels)
