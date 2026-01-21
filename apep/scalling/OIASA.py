from pathlib import Path
import sys
sys.path.extend([
    str(Path(__file__).resolve().parents[1]),
    str(Path(__file__).resolve().parents[2]),
])
import numpy as np
from geopy.distance import geodesic


#############################################################################################
#  Oblique Ionogram Automatric Scaling Algorithms (OIASA)
#  Cite: Ippolito, A., C. Scotto, D. Sabbagh, V. Sgrigna, and P. Maher (2016), 
#        A procedure for the reliability improvement of the oblique ionograms automatic 
#        scaling algorithm, Radio Sci., 51, 454–460, doi:10.1002/2015RS005919. 
#############################################################################################

def get_ground_distance(rx, tx):
    return geodesic(rx, tx).km

def get_theta(D, R=6371):
    return np.rad2deg(D/(R))/2

def get_curvature_correction(theta, R=6371):
    return R*(1-np.cos(np.deg2rad(theta)))

def get_virtual_height(p, c, phi):
    return (p/(2*np.sin(np.deg2rad(phi)))) - c

def get_fv(fo, phi):
    return fo*np.cos(np.deg2rad(phi))

def get_phi(D, p):
    return np.rad2deg(np.arcsin(D/p))