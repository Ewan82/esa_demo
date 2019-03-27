import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import datetime as dt
from matplotlib.dates import DateFormatter


def find_nearest(array, value):
    """
    Find nearest value in an array
    :param array: array of values
    :param value: value for which to find nearest element
    :return: nearest value in array, index of nearest value
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def find_nearest_idx_tol(array, value, tol=dt.timedelta(days=1.)):
    """
    Find nearest value in an array
    :param array: array of values
    :param value: value for which to find nearest element
    :return: nearest value in array, index of nearest value
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    if abs(array[idx] - value) <= tol:
        ret_val = idx
    else:
        ret_val = np.nan
    return ret_val


def plot_var_paper(var, _dir='ret_code', axes=None, point='508_med'):
    post = nc.Dataset(_dir+'/controlvector_post.nc', 'r')
    pri = nc.Dataset(_dir + '/controlvector_prior.nc', 'r')
    field_laican = mlab.csv2rec('data/field_obs/mni_lai_canht_field_'+ point + '.csv', comments='%')
    field_sm = mlab.csv2rec('data/field_obs/mni_sm_field_' + point + '.csv', comments='%')
    if axes is None:
        fig, ax = plt.subplots()
        ret_val = fig, ax
    else:
       ax = axes
       ret_val = ax
    var_dict = {'lai': r'Leaf area index (m$^{2}$ m$^{-2}$)', 'canht': 'Canopy height (m)',
                'sm': r'Soil moisture (m$^{3}$ m$^{-3}$)'}
    sat_times = nc.num2date(post.variables['time'][:], post.variables['time'].units)
    #ax.plot(times, np.array([np.nan]*len(times)), 'X')
    if var == 'sm':
        t_idx = np.array([find_nearest(field_sm['_date'], x)[1] for x in sat_times])
        field_times = field_sm['_date'][t_idx]
        field_ob = field_sm['sm'][t_idx]
    else:
        #idx = np.where((field_laican['_date'] < sat_times[-1].date()) & (field_laican['_date'] > sat_times[0].date()))
        field_times = field_laican['_date'][:]
        field_times = np.array([dt.datetime.combine(x,dt.datetime.min.time()) for x in field_times])
        field_ob = field_laican[var][:]
        t_idx = np.array([find_nearest_idx_tol(field_times, x, tol=dt.timedelta(days=2))
                          for x in sat_times])
        t_idx = t_idx[np.isnan(t_idx) == False]
        t_idx = np.array([int(x) for x in t_idx])
        field_times = field_laican['_date'][t_idx]
        field_ob = field_laican[var][t_idx]
    ax.errorbar(sat_times[pri.variables['sim_typ'][:] == 9], pri.variables[var][pri.variables['sim_typ'][:] == 9],
                # yerr=pri.variables[var+'_unc'][pri.variables['sim_typ'][:] == 1],
                fmt='X', label='prior S1', color='b', alpha=0.9)
    ax.errorbar(sat_times[pri.variables['sim_typ'][:] == 34], pri.variables[var][pri.variables['sim_typ'][:] == 34],
                #yerr=post.variables[var+'_unc'][post.variables['sim_typ'][:] == 2],
                fmt='X', label='prior S2', color='g', alpha=0.9)
    ax.errorbar(sat_times[post.variables['sim_typ'][:] == 9], post.variables[var][post.variables['sim_typ'][:] == 9],
                #yerr= pri.variables[var+'_unc'][pri.variables['sim_typ'][:] == 1],
                fmt='o', label='retrieval output S1', color='b', alpha=0.9)
    ax.errorbar(sat_times[post.variables['sim_typ'][:] == 34], post.variables[var][post.variables['sim_typ'][:] == 34],
                #yerr=post.variables[var+'_unc'][post.variables['sim_typ'][:] == 2],
                fmt='o', label='retrieval output S2', color='g', alpha=0.9)
    if var == 'sm':
        ax.set_ylim([0, 0.5])
    elif var == 'lai':
        ax.set_ylim([0, 8.0])
    ax.plot(field_times, field_ob, '*', label='Field obs', color='k')
    ax.set_xlabel('Date')
    ax.set_ylabel(var_dict[var])
    if axes is None:
        fig.autofmt_xdate()
    plt.legend(frameon=True, fancybox=True, framealpha=0.5)
    return ret_val
