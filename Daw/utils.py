import numpy as np
from matplotlib.patches import Ellipse

###############################################################
#
###############################################################
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

###############################################################
#
###############################################################
def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

###############################################################
#
###############################################################
def draw_ellipse(alpha_mean, alpha_std, beta_mean, beta_std, color, angle):
    for j in range(1, 2):
        ell1 = Ellipse(xy=(beta_mean, alpha_mean),
                      width=beta_std*j*2, height=alpha_std*j*2,
                      angle=angle)
        ell1.set_facecolor('none')
        ell1.set_edgecolor(color)

    return ell1
