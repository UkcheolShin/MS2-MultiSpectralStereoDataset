import numpy as np
from imageio import imread
from scipy.io import loadmat
import matplotlib as mpl
import matplotlib.cm as cm

def load_as_float_img(path):
    img =  imread(path).astype(np.float32)
    if len(img.shape) == 2: # for NIR and thermal images
        img = np.expand_dims(img, axis=2)
    return img

def load_as_float_depth(path):
    if 'png' in path:
        depth =  np.array(imread(path).astype(np.float32))
    elif 'npy' in path:
        depth =  np.load(path).astype(np.float32)
    elif 'mat' in path:
        depth =  loadmat(path).astype(np.float32)
    return depth

# Tempearture to raw thermal radiation value
# For the parameters (R,B,F,O), we use default values of FLIR A65C camera
def Celsius2Raw(celcius_degree):
    R = 380747
    B = 1428
    F = 1
    O = -88.539
    raw_value = R / (np.exp(B / (celcius_degree + 273.15)) - F) + O;
    return raw_value

# Raw thermal radiation value to tempearture 
def Raw2Celsius(Raw):
    R = 380747
    B = 1428
    F = 1
    O = -88.539
    Celsius = B / np.log(R / (Raw - O) + F) - 273.15;
    return Celsius

def visualize_disp_as_numpy(disp, cmap='jet'):
    """
    Args:
        data (HxW): disp data
        cmap: color map (inferno, plasma, jet, turbo, magma, rainbow)
    Returns:
        vis_data (HxWx3): disp visualization (RGB)
    """

    disp = disp.cpu().numpy()
    disp = np.nan_to_num(disp)  # change nan to 0

    vmin = np.percentile(disp[disp!=0], 0)
    vmax = np.percentile(disp[disp!=0], 95)

    normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap=cmap)
    vis_data = (mapper.to_rgba(disp)[:, :, :3] * 255).astype(np.uint8)
    return vis_data

def visualize_depth_as_numpy(depth, cmap='jet', is_sparse=True):
    """
    Args:
        data (HxW): depth data
        cmap: color map (inferno, plasma, jet, turbo, magma, rainbow)
    Returns:
        vis_data (HxWx3): depth visualization (RGB)
    """

    x = depth.cpu().numpy()
    x = np.nan_to_num(x)  # change nan to 0

    inv_depth = 1 / (x + 1e-6)

    if is_sparse:
        vmax = 1/np.percentile(x[x!=0], 5)
    else:
        vmax = np.percentile(inv_depth, 95)

    normalizer = mpl.colors.Normalize(vmin=inv_depth.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap=cmap)
    vis_data = (mapper.to_rgba(inv_depth)[:, :, :3] * 255).astype(np.uint8)
    if is_sparse:
        vis_data[inv_depth>vmax] = 0
    return vis_data