import numpy as np
import pickle
import glob
import matplotlib.pyplot as plt
import keras.backend as keras
import tensorflow as tf
from tqdm import tqdm_notebook
import skimage

def zero_lower_data(data, zrange):
    '''
    Specific function to make values in a range of the INS spectrum set to zero
    Args:
        data : array of shape (ndata, 20, 200, 1)
        zrange : the range over which to set to zero (list)
    Returns:
        The data with zeros in the range
    '''
    for x in data:
        for i in zrange:
            x[:, i, 0] = 0
    return data


def build_decay_map(image, decay_factor=0.02):
    '''
    Makes a map of values that decay exponentially from 1 along the x-axis
    Function e^(-decay_factor * x)
    Args:
        image: a 3D array (height, width, channels)
        decay_factor: the factor in the exponent of decay
    Returns:
        decay_map: a 3D array of the same dimenstions as image, with the 
        decaying factor across the width
    '''

    xmax = len(image[0])
    decay_map = np.zeros(image.shape)
    for i in range(xmax):
        decay_map[:, i, :] = decay_map[:, i, :] + exp_fn(i, 1, decay_factor)
    return decay_map

def apply_decay_map(data, decay_map):
    '''
    Apply the decay map built by `build_decay_map` to a set of images
    Args:
        data: the list of images, each one must have the same dimensions as the map
        decay_map: the map built by `build_decay_map`
    Returns:
        An array of images with the decay factor applied
    '''
    new_data = []
    for image in data:
        image_new = image * decay_map
        new_data.append(image_new)
    return np.array(new_data)

def exp_fn(x, a, lam):
    return a * np.exp(-lam * x)

def norm_data(img):
    maxval = np.max(img)
    minval = np.min(img)
    return (img - minval)/(maxval - minval)

def clip_q_range(spec, lower_q, upper_q):
    '''
    Clip the range of q values to sample over
    Args:
        spec the spectrum to process [(qpos), (intensities)]
        lower and upper values of q to consider
    Returns
        [(qpos), (intensities)]
    '''
    clip_qs = []
    for point in spec:
        if (point[0][0] >= lower_q) & (point[0][0] <= upper_q) & \
        (point[0][1] >= lower_q) & (point[0][1] <= upper_q):
            clip_qs.append(point)
    return clip_qs
    

def load_spectra(filename, start=0, end=10, lower_q=-2., upper_q=2.):
    ''' 
    A function to read in the data from the files
    Args:
        filename
        start: the energy from which to start collecting
        end: the energy at which to finish collecting
        The data is in the form
            [(qpos), (intensities)]
            qpos is a tuple of the q location, these are sampled from -1 to 1 in 20 steps
            intemsities are 10 energy bins from 2 - 50 meV with band intensity in that bin 
    '''
    
    inpickle = open(filename, 'rb')
    try:
        full_spectrum = pickle.load(inpickle)
    except:
        print('Found incomplete file %s; skipping this entry' % filename)
        return []
    full_spectrum = clip_q_range(full_spectrum, lower_q, upper_q)
    qrange = int(np.sqrt((np.array(full_spectrum).shape[0])))
    q_point_spectra = [f[1][0][start:end] for f in full_spectrum]
    eners = []
    try:
        new = np.reshape(q_point_spectra, (qrange, qrange, start - end))
    except:
        print('Found incommensurate shape %s; skipping this entry' % filename)
        return []
    
    where_are_NaNs = np.isnan(q_point_spectra)
    q_point_spectra = np.asarray(q_point_spectra)
    q_point_spectra[where_are_NaNs] = 0
    
    for i in range(end - start):
        eners.append(np.array([q[i] for q in q_point_spectra]).reshape(qrange, qrange))
    new_qs = np.zeros(shape = (qrange, (end - start) * qrange))
    for i in range(10):
        for j in range(qrange):
            for k in range(qrange):
                new_qs[j,  k + i * qrange] = eners[i][j, k]
    if(np.array(new_qs).shape != (20, (end - start)* 20)):
        print('Processed spectrum %s incorrect shape, skipping' % filename)
        return []
    return new_qs

def mean_of_all_images(images):
    '''Returns the mean over a full list of images'''
    mean = []
    for entry in images:
        mean.append(np.mean(entry.flatten()))
    return np.mean(mean)

def max_of_all_images(images):
    '''Returns the max over a full list of images'''
    maxi = 0
    for entry in images:
        if np.max(entry) > maxi:
            maxi = np.max(entry)
    return maxi

def min_of_all_images(images):
    '''Returns the max over a full list of images'''
    mini = 0
    for entry in images:
        if np.min(entry) > mini:
            mini = np.min(entry)
    return mini

def global_average_pooling(x):
    return keras.mean(x, axis = (1, 2))
    
def global_average_pooling_shape(input_shape):
     return [input_shape[0], input_shape[3]]

def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer


def load_ins_data(data_path, number=1):
    goodenough = [name for name in glob.glob(data_path +'*pkl')]
    number_of_samples = number # The number of sample spectra to take
    range_lower = 0
    range_upper = 10
    lower_q = -2
    upper_q = 2
    goodenough_list = []
    ge_spectra = []
    for i, f in enumerate(goodenough[:number_of_samples]):
        spec = load_spectra(f, start=range_lower, end=range_upper, lower_q=lower_q, upper_q=upper_q)
        if type(spec) == np.ndarray and max(spec.flatten()) < 300:
            ge_spectra.append(spec)
# Normalise the spectra
    ge_max = max_of_all_images(ge_spectra)
    ge_min = min_of_all_images(ge_spectra)
    for item in ge_spectra:
        goodenough_list.append(('goodenough', (item - ge_min)/(ge_max - ge_min)))
    return goodenough_list

def histogram_data(data, lower_e=2, upper_e=50, steps_e=10, bins=20,
                  lower_q=-2., upper_q=2., normalise=False):
    '''
    Takes in data in E/Q space - 2D in Q plus E dimension.
    (i)   Sorts the data into energy ranges based on lower_e, upper_e and steps_e.
    (ii)  Dicards any data outside upper and lower q.
    (iii) Makes 2D histograms of data in Q-space for each energy range; weighted by a given property
    (iv)  Stacks the 2D histograms to make a single 2D dataset, with energy slices side-by-side
    '''

    data_bins = []
    for i in np.arange(lower_e, upper_e, (upper_e - lower_e)/steps_e):
        data_bins.append(data[(data.energy > i) & (data.energy < i + (upper_e - lower_e)/steps_e)
                             & (data.qh > lower_q) & (data.qh < upper_q)
                             & (data.qk > lower_q) & (data.qk < upper_q)])
    
    for i in range(0, len(data_bins)):
        a = data_bins[i]
        if i == 0:
            hists, x, y, im = plt.hist2d(a.qh, a.qk, weights=a.proc, bins=bins)
            if normalise:
                hists = hists / np.mean(hists)
        else:
            tmp, x, y, im = plt.hist2d(a.qh, a.qk, weights=a.proc, bins=bins)
            if normalise:
                hists = np.hstack((hists, tmp / np.mean(tmp)))
            else:
                hists = np.hstack((hists, tmp))
    return hists

def inverse_boltzman(df, i, fact=8.617e0):
    '''
    In the original PRL the expt data was weighted with an inverse Boltzman factor, to ensure that population 
    effects do not affect the intensities of the signals. 
    Args:
        df: The list of dataframes for different incident neutron energies
        i: the index of the dataframe to process
        factor: the weighting factor to use
    '''
    return df[i][1]['Count'].apply(lambda x: (x/fact)/(1 - np.exp(-x/fact)))

def gauss_noise(img, mean=0, var=0.1):
    row = img.shape[0]
    col = img.shape[1]
    ch = img.shape[2]
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row, col, ch)
    return img + gauss

def poiss_noise(img, factor=2):
    vals = len(np.unique(img))
    vals = factor ** np.ceil(np.log2(vals))
    return np.random.poisson(img * vals) / float(vals)

def comp_noise(img, mean=0, var=0.00001, factor=1.6):
    mean = np.mean(img)
    noised = poiss_noise(img, factor=factor)
    noised = gauss_noise(noised, mean, var=var)
    return noised

def add_detector_pixels(expt, simulations, thresh=1e1):
    '''
    Extracts the low value (essentially zero signal) pixels from an experimental spectrum
    Sets the corresponding elements to zero in the simulated spectra fed in

    '''
    e = expt.reshape(simulations[0].shape)
    where_are_zeros = np.argwhere(e <= thresh)
    for s in simulations:
        for z in where_are_zeros:
            s[z[0], z[1], z[2]] = 0.
    return simulations

def noise_and_mask(imgs, expt, var=0.0002, factor=1.4, thresh=7e-2):
    expt = norm_data(expt)
    new_imgs = []
    for img in imgs:
        img = norm_data(img)
        img = comp_noise(img, np.mean(img), var, factor)
        new_imgs.append(np.clip(skimage.gaussian_filter(img, sigma=0.75), 0., None))
    new_imgs = add_detector_pixels(expt, new_imgs, thresh)
    return new_imgs

def make_activation_model(model):
    """Make an 'activation model' for the pre-trained model
    
    Takes an existing, pre-trained model and finds all convolution layers.
    Then creates a new model where every convolutional layer is also an 
    output layer.
    
    Args:
        model: a pre-trained model with convolutional layers
    Returns:
        an activation model where all convolutional layers are also output layers
    """
    layers = [layer for layer in model.layers if 'conv' in layer.name]
    layer_outputs = [layer.output for layer in layers]
    layer_names = [layer.name for layer in layers]
    
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
    return activation_model

def maximize_filter_activation(input_img, activation_model, layer_index, filter_index, n_iter=20, step=1):
    """Maximize the activation of a given filter
    
    Based on the example code for Keras by F. Chollet:
    https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
    
    Args:
        input_img: input image to maxmize the activation for. This is usually gaussian random noise. 
        activation_model: an activation model to use to sample filters from
        layer: layer to maximize given filter for.
        layer_index: index of the layer to maximize.
        
    Returns:
        Maximized output for the given filter of shape (num filters, h, w, channels)
        loss for that filter
    """
    input_img_data = tf.constant(input_img, dtype=tf.float32)
    for i in range(n_iter):
        with tf.GradientTape() as tape:
            tape.watch(input_img_data)
            loss = tf.math.reduce_mean(activation_model(input_img_data)[layer_index][:, :, :, filter_index])
            # compute the gradient of the input picture wrt this loss
            grads = tape.gradient(loss, input_img_data)
            # normalization trick: we normalize the gradient
            grads /= (tf.math.sqrt(tf.math.reduce_mean(tf.math.square(grads))) + 1e-5)
            input_img_data += (grads * step)

    return input_img_data, loss

def max_filter_samples(input_img, activation_model, layer_index, n_samples=5, **kwargs):
    """Get the filters that respond the most to input signals. We get this by seleting the filters with the lowest
    loss as returned by maximize_filter_activation    
    Args:
        input_img: input image to maxmize the activation for. This is usually gaussian random noise. 
        activation_model: an activation model to use to sample filters from
        layer_index: index of the layer to maximize filters for.
        n_select: the top number of filters to return
        
    Returns:
        list of maximized activations for a random sample of filters.
    """
    n_filters = activation_model.output_shape[layer_index][-1]
    indices = range(n_filters)
    outputs = []
    for filter_index in indices:
        output = maximize_filter_activation(input_img, activation_model, layer_index, filter_index, **kwargs)
        outputs.append([output[0].numpy(), output[1].numpy()])
    outputs = sorted(outputs, key=lambda x: x[1], reverse=True)
    outputs = [o[0] for o in outputs]
    n_samples = n_filters if n_filters < n_samples else n_samples
    return outputs[:n_samples]

def random_filter_samples(input_img, activation_model, layer_index, n_samples=10, **kwargs):
    """Maximize the activation for a random sample of filters
    
    Args:
        input_img: input image to maxmize the activation for. This is usually gaussian random noise. 
        activation_model: an activation model to use to sample filters from
        layer_index: index of the layer to maximize filters for.
        
    Returns:
        list of maximized activations for a random sample of filters.
    """
    n_filters = activation_model.output_shape[layer_index][-1]
    n_samples = n_filters if n_filters < n_samples else n_samples
    indices = np.random.choice(n_filters, n_samples, replace=False)
    outputs = []
    for filter_index in indices:
        output = maximize_filter_activation(input_img, activation_model, layer_index, filter_index, **kwargs)
        outputs.append(output[0].numpy())
    return outputs

def sample_model_filters(model, input_img, mode='random', **kwargs):
    """Sample filters from a model
    
    This will randomly sample filters from the model and
    maximise the activation of that filter using the given input image/
    
    Args:
        model: input model to sample filters from
        input_img: the input image to use to maximise filter activation.
            Must match the input shape expected by the model.
    
    Returns:
        list of samples of maximized input for filters form each layer of the model
    """
    activation_model = make_activation_model(model)
    model_layer_outputs = [ ]
    for layer_index in tqdm_notebook(range(len(activation_model.output_shape))):
        if mode == 'random':
            output = random_filter_samples(input_img, activation_model, layer_index, **kwargs)
        elif mode == 'max':
            output = max_filter_samples(input_img, activation_model, layer_index, **kwargs)
        model_layer_outputs.append(output)
    return model_layer_outputs

def extract_activations(model, layer_name, input_img):
    '''
    Extract the activation values for a given layer of a model with a certain input image.
    Args:
        model: a tf model
        layer_name: string
        input_img: the input image
    '''
    layer = get_output_layer(model, layer_name)
    get_values = tf.keras.backend.function([model.layers[0].input],
                                          [layer.output])
    return np.array(get_values(input_img)).squeeze()

def sample_activation_map(model, layer_name, sample, max_maps, class_idx=0):
    '''
    The idea here is to build the images that maximise each filter in the final layer (I), 
    then multiply these I by the activation value of that filter for a given input (A) - 
    we then take the mean of A×I across a sample of images corresponding to a given class. 
    Args:
        model: a tf model
        layer_name: string
        sample: an input image
        max_maps: the filter maximisation images genetated by layer_max_maps
        class_idx: the class being looked for
    '''
    img_activations = extract_activations(model, layer_name, sample)
    act_map = np.zeros(shape=max_maps[0][0].numpy().shape)
    class_weights = model.layers[-1].get_weights()[0]
    for i in range(len(img_activations)):
        act_map += img_activations[i] * max_maps[i][0].numpy() \
                   * class_weights[i][class_idx]
    return act_map

def layer_max_maps(model, layer_index, input_img, **kwargs):
    '''
    This function builds the maps that maximise the filter outputs for all the filters in layer_index
    of model.
    Args:
        model: a tf model
        layer_name: string
        input_img: an input image
    '''
    active_model = make_activation_model(model)
    max_maps = []
    n_filters = model.layers[layer_index].output_shape[-1]
    for i in range(n_filters):
        max_map = maximize_filter_activation(input_img, active_model, layer_index-1,  ### NB remove this magic 1
                                        filter_index=i, n_iter=20, step=1)
        max_maps.append(max_map)
    return max_maps

def integrate_L(simulated_image_full, q_pos):
    enst = np.vstack([s['en'] for s in simulated_image_full])
    inst = np.vstack([s['i'] for s in simulated_image_full])
    simulated_image = []
    for hk in q_pos[:1600,:]: #np.vstack({tuple(q[:2]) for q in q_pos}):
        idx = (q_pos[:,0] == hk[0]) * (q_pos[:,1] == hk[1])
        simulated_image.append({'q':[hk[0], hk[1], 0], 'en':enst[idx,:].flatten(), 'i':inst[idx,:].flatten()})
    return simulated_image
