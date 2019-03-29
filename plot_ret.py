import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import datetime as dt


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


def plot_var(var, _dir='ret_code', axes=None, point='508_med'):
    """
    Function which plots the output of the retrieval tool, including site level observations for comparison.
    :param var: which variable to plot, lai, sm or canht (str)
    :param _dir: directory of retrieval tool output (str)
    :param axes: axes to plot on, if None new axes generated (obj)
    :param point: which point to plot field data from (str)
    :return: figure and axes of plot (obj)
    """
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
    if var == 'sm':
        t_idx = np.array([find_nearest(field_sm['_date'], x)[1] for x in sat_times])
        field_times = field_sm['_date'][t_idx]
        field_ob = field_sm['sm'][t_idx]
    else:
        field_times = field_laican['_date'][:]
        field_times = np.array([dt.datetime.combine(x,dt.datetime.min.time()) for x in field_times])
        t_idx = np.array([find_nearest_idx_tol(field_times, x, tol=dt.timedelta(days=2))
                          for x in sat_times])
        t_idx = t_idx[np.isnan(t_idx) == False]
        t_idx = np.array([int(x) for x in t_idx])
        field_times = field_laican['_date'][t_idx]
        field_ob = field_laican[var][t_idx]
    ax.errorbar(sat_times[pri.variables['sim_typ'][:] == 9], pri.variables[var][pri.variables['sim_typ'][:] == 9],
                # yerr=pri.variables[var+'_unc'][pri.variables['sim_typ'][:] == 1],
                fmt='x', label='prior S1', color='b', alpha=0.7)
    ax.errorbar(sat_times[pri.variables['sim_typ'][:] == 34], pri.variables[var][pri.variables['sim_typ'][:] == 34],
                #yerr=post.variables[var+'_unc'][post.variables['sim_typ'][:] == 2],
                fmt='x', label='prior S2', color='g', alpha=0.7)
    ax.errorbar(sat_times[post.variables['sim_typ'][:] == 9], post.variables[var][post.variables['sim_typ'][:] == 9],
                #yerr= pri.variables[var+'_unc'][pri.variables['sim_typ'][:] == 1],
                fmt='o', label='retrieval output S1', color='b', alpha=0.7)
    ax.errorbar(sat_times[post.variables['sim_typ'][:] == 34], post.variables[var][post.variables['sim_typ'][:] == 34],
                #yerr=post.variables[var+'_unc'][post.variables['sim_typ'][:] == 2],
                fmt='o', label='retrieval output S2', color='g', alpha=0.7)
    post.close()
    pri.close()
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


def plot_var_season(var, _dir='ret_code', axes=None, point='508_med', field_obs_on=True):
    post = nc.Dataset(_dir+'/controlvector_post.nc', 'r')
    pri = nc.Dataset(_dir + '/controlvector_prior.nc', 'r')
    field_laican = mlab.csv2rec('/home/users/if910917/projects/ret_tool_subprocess/state_files/mni_lai_canht_field_'+
                                point+'.csv', comments='%')
    field_sm = mlab.csv2rec('/home/users/if910917/projects/ret_tool_subprocess/state_files/mni_sm_field_'+point+'.csv',
                            comments='%')
    if axes is None:
        fig, ax = plt.subplots()
        ret_val = fig
    else:
        ax = axes
        ret_val = ax
    var_dict = {'lai': r'Leaf area index (m$^{2}$ m$^{-2}$)', 'canht': 'Canopy height (m)',
                'sm': r'Soil moisture (m$^{3}$ m$^{-3}$)'}
    sat_times = nc.num2date(post.variables['time'][:], post.variables['time'].units)
    if var == 'sm':
        t_idx = np.array([find_nearest(field_sm['_date'], x)[1] for x in sat_times])
        field_times = field_sm['_date'][t_idx]
        field_ob = field_sm['sm'][t_idx]
    else:
        field_times = field_laican['_date'][:]
        field_ob = field_laican[var][:]
    ax.errorbar(sat_times[pri.variables['sim_typ'][:] == 9], pri.variables[var][pri.variables['sim_typ'][:] == 9],
                # yerr=pri.variables[var+'_unc'][pri.variables['sim_typ'][:] == 1],
                fmt='x', label='prior S1', color='b', alpha=0.7)
    ax.errorbar(sat_times[pri.variables['sim_typ'][:] == 34], pri.variables[var][pri.variables['sim_typ'][:] == 34],
                #yerr=post.variables[var+'_unc'][post.variables['sim_typ'][:] == 2],
                fmt='x', label='prior S2', color='g', alpha=0.7)
    ax.errorbar(sat_times[post.variables['sim_typ'][:] == 9], post.variables[var][post.variables['sim_typ'][:] == 9],
                #yerr= post.variables[var+'_unc'][post.variables['sim_typ'][:] == 1],
                fmt='o', label='Retrieval output S1', color='b', alpha=0.7)
    ax.errorbar(sat_times[post.variables['sim_typ'][:] == 34], post.variables[var][post.variables['sim_typ'][:] == 34],
                #yerr=post.variables[var+'_unc'][post.variables['sim_typ'][:] == 2],
                fmt='o', label='Retrieval output S2', color='g', alpha=0.7)
    if var == 'sm':
        ax.set_ylim([0, 0.5])
    elif var == 'lai':
        ax.set_ylim([0, 8.0])
    if field_obs_on is True:
        ax.plot(field_times, field_ob, '*', label='Field obs', color='k')
    ax.set_xlabel('Date')
    ax.set_ylabel(var_dict[var])
    if axes is None:
        fig.autofmt_xdate()
    plt.legend(frameon=True, fancybox=True, framealpha=0.5)

    if var == 'sm':
        t_idx = np.array([find_nearest_idx_tol(field_sm['_date'], x, tol=dt.timedelta(seconds=60*60*3))
                          for x in sat_times])
        t_idx = t_idx[np.isnan(t_idx) == False]
        t_idx = np.array([int(x) for x in t_idx])
        field_times = field_sm['_date'][t_idx]
        field_ob = field_sm['sm'][t_idx]
        field_times = field_times[np.isnan(field_ob) == False]
        field_ob = field_ob[np.isnan(field_ob) == False]
        ret_t_idx = np.array([find_nearest_idx_tol(sat_times, x, tol=dt.timedelta(seconds=60*60*3))
                          for x in field_times])
        post_obs = post.variables[var][ret_t_idx]
    else:
        field_times = field_laican['_date'][:]
        field_times = np.array([dt.datetime.combine(x,dt.datetime.min.time()) for x in field_times])
        field_ob = field_laican[var][:]
        t_idx = np.array([find_nearest_idx_tol(sat_times, x, tol=dt.timedelta(days=2))
                          for x in field_times])
        t_idx = t_idx[np.isnan(t_idx) == False]
        t_idx = np.array([int(x) for x in t_idx])
        post_obs = post.variables[var][t_idx]
        sat_times = sat_times[t_idx]
        ret_t_idx = np.array([find_nearest_idx_tol(field_times, x, tol=dt.timedelta(days=2))
                          for x in sat_times])
        field_ob = field_ob[ret_t_idx]

    #innov = [((field_ob[i] - np.mean(field_ob)) - (post_obs[i] - np.mean(post_obs)))**2 for i in xrange(len(post_obs))]
    #pos_ubrmse = np.sqrt(np.sum(innov) / len(post_obs))
    #innov = [((field_ob[i] - np.mean(field_ob)) - (pri_obs[i] - np.mean(pri_obs)))**2 for i in xrange(len(post_obs))]
    #pri_ubrmse = np.sqrt(np.sum(innov) / len(post_obs))

    pos_corrc = np.corrcoef(post_obs, field_ob)[0,1]
    print('correlation: '+str(pos_corrc))

    innov = [(post_obs[i] - field_ob[i])**2 for i in xrange(len(post_obs))]
    pos_rmse = np.sqrt(np.sum(innov) / len(post_obs))
    print('rmse: '+str(pos_rmse))

    post.close()
    pri.close()
    return ret_val