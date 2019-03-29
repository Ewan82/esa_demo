import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from copy import copy
import signaturesimulator as ss


S2_band_labels = ['1', '2', '3', '4', '5', '6', '7', '8', '8a', '9', '10', '11', '12']


def run_smac_fwd(refl, geom, pressure=1013, uo3=0.3, uh2o=3.0, aot=0.1):
    """
    Factory function forward modelling signature simulator outputted reflectances to the top of atmosphere using the
    SMAC atmospheric correction model.
    :param refl: array of reflectance values for the Senintel 2 bands
    :param geom: geomtry class from the signature simulator
    :param pressure: pressure at pixel altitude
    :param uo3: ozone content (cm)  0.3 cm = 300 Dobson Unit
    :param uh2o: water vapour (g cm-2)
    :param aot: aerosol optical thickness at 550 nm
    :return:
    """
    len_refl = len(refl)
    bands = {'1': [0, np.zeros(len_refl)], '2': [1, np.zeros(len_refl)], '3': [2, np.zeros(len_refl)],
             '4': [3, np.zeros(len_refl)], '5': [4, np.zeros(len_refl)], '6': [5, np.zeros(len_refl)],
             '7': [6, np.zeros(len_refl)], '8': [7, np.zeros(len_refl)], '8a': [8, np.zeros(len_refl)],
             '9': [9, np.zeros(len_refl)], '10': [10, np.zeros(len_refl)], '11': [11, np.zeros(len_refl)],
             '12': [12, np.zeros(len_refl)], }
    for band in bands.keys():
        coefs_f = 'COEFS/Coef_S2A_CONT_B' + band + '.dat'
        coeffs = smac.coeff(coefs_f)
        bands[band][1] = smac.smac_dir(refl[:, bands[band][0]], geom.sza[0], geom.saa[0], geom.vza[0], geom.vaa[0],
                                       pressure, aot, uo3, uh2o, coeffs)
    return bands



def plot_class_var(sim_class, sim_var, y_lab=None, line_type='-'):
    """Plot specified variable.

    :param var: Class attribute variable as list.
    :type var: list
    :param y_lab: Label for Y-axis.
    :type y_lab: str
    :return: Figure.
    :rtype: object
    """
    plt.plot(sim_class.get_geom.date_utc, sim_var, line_type)
    plt.ylabel(y_lab)
    plt.xlabel('Date')
    return 'plotted'


def plot_refl(sim_class, band_idx, line_type='-'):
    """Plot specified variable.

    :param var: Class attribute variable as list.
    :type var: list
    :param y_lab: Label for Y-axis.
    :type y_lab: str
    :return: Figure.
    :rtype: object
    """
    plt.plot(sim_class.get_geom.date_utc, np.array(sim_class.spectra.refl)[:, band_idx], line_type)
    plt.ylabel('Band ' + S2_band_labels[band_idx] + ' reflectance')
    plt.xlabel('Date')
    return 'plotted'


def plot_ndvi(sim_class, line_type='-'):
    """Plot specified variable.

    :param var: Class attribute variable as list.
    :type var: list
    :param y_lab: Label for Y-axis.
    :type y_lab: str
    :return: Figure.
    :rtype: object
    """
    refl = np.array(sim_class.spectra.refl)
    plt.plot(sim_class.get_geom.date_utc, (refl[:, 7]-refl[:, 3])/(refl[:, 7]+refl[:, 3]), line_type)
    plt.ylabel('NDVI')
    plt.xlabel('Date')
    return 'plotted'


def sentinel2_wavl(mission="a"):
    sim=ss.Simulator()
    if mission == "a":
        fname = sim.dir_path+"/data/srf/s2a.srf"
    else:
        fname = sim.dir_path+"/data/srf/s2b.srf"
    wavl=[]
    for n in xrange(0, 13):
        tmp = np.loadtxt(fname, delimiter=" ", usecols=(n))
        idx = np.argmax(tmp)
        wavl.append(400+idx)
    wavl = np.array(wavl)
    wavl_sub = np.delete(wavl, [0,9,10])
    return wavl_sub