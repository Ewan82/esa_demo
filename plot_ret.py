import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.mlab as mlab
import signaturesimulator as ss
import seaborn as sns
import datetime as dt
from matplotlib.dates import DateFormatter
import taylordiagram as td


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


def plot_var(var, _dir='ret_code', axes=None, point='508_med', field_obs_on=True):
    prior_val = nc.Dataset(_dir+'/controlvector_prior.nc', 'r')
    post = nc.Dataset(_dir+'/controlvector_post.nc', 'r')
    if point == '508_farmyard':
        point = '508_med'
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
    ax.plot(sat_times, prior_val.variables[var][:], '+', label='Prior')
    #ax.plot(times, np.array([np.nan]*len(times)), 'X')
    if var == 'sm':
        t_idx = np.array([find_nearest(field_sm['_date'], x)[1] for x in sat_times])
        print t_idx
        field_times = field_sm['_date'][t_idx]
        field_ob = field_sm['sm'][t_idx]
    else:
        field_times = field_laican['_date'][:]
        field_ob = field_laican[var][:]
    ax.errorbar(sat_times[post.variables['sim_typ'][:] == 9], post.variables[var][post.variables['sim_typ'][:] == 9],
                #yerr= post.variables[var+'_unc'][post.variables['sim_typ'][:] == 1],
                fmt='o', label='Retrieval output S1')
    ax.errorbar(sat_times[post.variables['sim_typ'][:] == 34], post.variables[var][post.variables['sim_typ'][:] == 34],
                #yerr=post.variables[var+'_unc'][post.variables['sim_typ'][:] == 2],
                fmt='o', label='Retrieval output S2')
    if var == 'sm':
        ax.set_ylim([0, 0.5])
    elif var == 'lai':
        ax.set_ylim([0, 8.0])
    elif var == 'canht':
        ax.set_ylim([0, 3.0])
    if field_obs_on is True:
        ax.plot(field_times, field_ob, '*', label='Field obs')
    ax.set_xlabel('Date')
    ax.set_ylabel(var_dict[var])
    if axes is None:
        fig.autofmt_xdate()
    plt.legend(frameon=True, fancybox=True, framealpha=0.5)
    return ret_val


def plot_refl_mod(_dir, point, axes=None):  # write a fn that plots obs and modelled refl for given point!
    if axes is None:
        fig, ax = plt.subplots()
        ret_val = fig
    else:
        ax = axes
        ret_val = ax
    pri = nc.Dataset(_dir+'/controlvector_prior.nc', 'r')
    post = nc.Dataset(_dir+'/controlvector_post.nc', 'r')
    post_times = nc.num2date(post.variables['time'][:], post.variables['time'].units)
    sat_obs = mlab.csv2rec('/home/users/if910917/projects/ret_tool_subprocess/S2/mni_s2_'+
                                point+'_b4b8.csv', comments='%')
    refls_pri = []
    refls_post = []
    sim = ss.Simulator(site_nml='/home/users/if910917/projects/ret_tool_subprocess/site.nml')
    for time in enumerate(sat_obs['_date']):
        t_idx = find_nearest(post_times, time[1])[1]
        sim.get_geom = sim.geom_default(date_utc=time[1], vza=sat_obs['vza'][time[0]], vaa=sat_obs['vaa'][time[0]],
                                        sza=sat_obs['sza'][time[0]],saa=sat_obs['saa'][time[0]])
        sim.get_land_state = sim.state_default(date_utc=time[1], lai=pri.variables['lai'][t_idx],
                                               canopy_ht=pri.variables['canht'][t_idx],
                                               soil_m=pri.variables['sm'][t_idx])
        sim.run_rt = sim.passive_optical
        sim.run()
        refls_pri.append(sim.spectra.refl[0])
        sim.get_land_state = sim.state_default(date_utc=time[1], lai=post.variables['lai'][t_idx],
                                           canopy_ht=post.variables['canht'][t_idx],
                                           soil_m=post.variables['sm'][t_idx])
        sim.run()
        refls_post.append(sim.spectra.refl[0])
    ax.plot(sat_obs['_date'],np.array(refls_pri)[:,3], '+', label='Band 4 prior')
    ax.plot(sat_obs['_date'], np.array(refls_post)[:, 3], 'o', label='Band 4 posterior')
    ax.plot(sat_obs['_date'], sat_obs['b4'], '*', label='Band 4 observations')
    ax.plot(sat_obs['_date'], np.array(refls_pri)[:, 7], '+', label='Band 8 prior')
    ax.plot(sat_obs['_date'], np.array(refls_post)[:, 7], 'o', label='Band 8 posterior')
    ax.plot(sat_obs['_date'], sat_obs['b8a'], '*', label='Band 8 observations')
    ax.set_xlabel('Date')
    ax.set_ylabel('Band reflectance')
    if axes is None:
        fig.autofmt_xdate()
    ax.legend(loc=0, frameon=True, fancybox=True, framealpha=0.5)
    ax.set_ylim([0.0, 0.6])
    return ret_val


def plot_refl(_dir, point, axes=None):  # write a fn that plots obs and modelled refl for given point!
    if axes is None:
        fig, ax = plt.subplots()
        ret_val = fig
    else:
        ax = axes
        ret_val = ax
    sat_obs = mlab.csv2rec('/home/users/if910917/projects/ret_tool_subprocess/S2/mni_s2_'+
                                point+'_b4b8.csv', comments='%')
    ax.plot(sat_obs['_date'], sat_obs['b4'], '*', label='Band 4 observations', color='r')
    ax.plot(sat_obs['_date'], sat_obs['b8a'], '*', label='Band 8 observations', color='g')
    ax.set_xlabel('Date')
    ax.set_ylabel('Band reflectance')
    if axes is None:
        fig.autofmt_xdate()
    ax.set_ylim([0.0, 0.6])
    ax.legend(loc=0, frameon=True, fancybox=True, framealpha=0.5)
    return ret_val


def plot_backscat(_dir, point, axes=None):  # write a fn that plots obs and modelled refl for given point!
    if axes is None:
        fig, ax = plt.subplots()
        ret_val = fig
    else:
        ax = axes
        ret_val = ax
    sat_obs = mlab.csv2rec('/home/users/if910917/projects/ret_tool_subprocess/S1/mni_s1_'+
                                point+'_hvvv.csv', comments='%')
    ax.plot(sat_obs['_date'], sat_obs['hv'], '*', label='HV observations', color='r')
    ax.plot(sat_obs['_date'], sat_obs['vv'], '*', label='VV observations', color='g')
    ax.set_xlabel('Date')
    ax.set_ylabel(r'Backscatter (m$^{2}$ m$^{-2}$)')
    if axes is None:
        fig.autofmt_xdate()
    ax.set_ylim([0.0, 0.15])
    ax.legend(loc=0, frameon=True, fancybox=True, framealpha=0.5)
    return ret_val


def plot_backscat_mod(_dir, point, axes=None):  # write a fn that plots obs and modelled refl for given point!
    if axes is None:
        fig, ax = plt.subplots()
        ret_val = fig
    else:
        ax = axes
        ret_val = ax
    pri = nc.Dataset(_dir+'/controlvector_prior.nc', 'r')
    post = nc.Dataset(_dir+'/controlvector_post.nc', 'r')
    post_times = nc.num2date(post.variables['time'][:], post.variables['time'].units)
    sat_obs = mlab.csv2rec('/home/users/if910917/projects/ret_tool_subprocess/S1/mni_s1_'+
                                point+'_hv.csv', comments='%')
    backscat_pri = []
    backscat_post = []
    sim = ss.Simulator(site_nml='/home/users/if910917/projects/ret_tool_subprocess/site.nml')
    for time in enumerate(sat_obs['_date']):
        t_idx = find_nearest(post_times, time[1])[1]
        sim.get_geom = sim.geom_default(date_utc=time[1], vza=sat_obs['vza'][time[0]], vaa=sat_obs['vaa'][time[0]],
                                        sza=sat_obs['sza'][time[0]],saa=sat_obs['saa'][time[0]])
        sim.get_land_state = sim.state_default(date_utc=time[1], lai=pri.variables['lai'][t_idx],
                                               canopy_ht=pri.variables['canht'][t_idx],
                                               soil_m=pri.variables['sm'][t_idx])
        sim.run_rt = sim.active_microwave
        sim.run()
        backscat_pri.append(sim.backscat.hv[0])
        sim.get_land_state = sim.state_default(date_utc=time[1], lai=post.variables['lai'][t_idx],
                                           canopy_ht=post.variables['canht'][t_idx],
                                           soil_m=post.variables['sm'][t_idx])
        sim.run_rt = sim.active_microwave
        sim.run()
        backscat_post.append(sim.backscat.hv)
    ax.plot(sat_obs['_date'],backscat_pri, '+', label='HV Prior')
    ax.plot(sat_obs['_date'], backscat_post, 'o', label='HV Posterior')
    ax.plot(sat_obs['_date'], sat_obs['hv'], '*', label='HV Observations')
    ax.set_xlabel('Date')
    ax.set_ylabel(r'Backscatter (m$^{2}$ m$^{-2}$)')
    if axes is None:
        fig.autofmt_xdate()
    ax.set_ylim([0.0, 0.15])
    ax.legend(loc=0, frameon=True, fancybox=True, framealpha=0.5)
    return ret_val
#def plot_backscat(__dir, point):   write a fn that plots obs and model backscat for given point!


def save_stats(var, _dir, point='508_mean', _all=True):
    pri = nc.Dataset(_dir+'/controlvector_prior.nc', 'r')
    post = nc.Dataset(_dir+'/controlvector_post.nc', 'r')
    field_laican = mlab.csv2rec('/home/users/if910917/projects/ret_tool_subprocess/state_files/mni_lai_canht_field_'+
                                point+'.csv', comments='%')
    field_sm = mlab.csv2rec('/home/users/if910917/projects/ret_tool_subprocess/state_files/mni_sm_field_'+point+'.csv',
                            comments='%')
    sat_times = nc.num2date(post.variables['time'][:], post.variables['time'].units)
    if var == 'sm':
        t_idx = np.array([find_nearest_idx_tol(field_sm['_date'], x, tol=dt.timedelta(seconds=60*60*3))
                          for x in sat_times])
        t_idx = t_idx[np.isnan(t_idx) == False]
        t_idx = np.array([int(x) for x in t_idx])
        #print t_idx
        field_times = field_sm['_date'][t_idx]
        field_ob = field_sm['sm'][t_idx]
        field_times = field_times[np.isnan(field_ob) == False]
        #print field_times
        field_ob = field_ob[np.isnan(field_ob) == False]
        ret_t_idx = np.array([find_nearest_idx_tol(sat_times, x, tol=dt.timedelta(seconds=60*60*3))
                          for x in field_times])
        #print t_idx
        post_obs = post.variables[var][ret_t_idx]
        pri_obs = pri.variables[var][ret_t_idx]
    else:
        field_times = field_laican['_date'][:]
        field_times = np.array([dt.datetime.combine(x,dt.datetime.min.time()) for x in field_times])
        field_ob = field_laican[var][:]
        t_idx = np.array([find_nearest_idx_tol(sat_times, x, tol=dt.timedelta(days=2))
                          for x in field_times])
        t_idx = t_idx[np.isnan(t_idx) == False]
        t_idx = np.array([int(x) for x in t_idx])
        post_obs = post.variables[var][t_idx]
        pri_obs = pri.variables[var][t_idx]
        sat_times = sat_times[t_idx]
        ret_t_idx = np.array([find_nearest_idx_tol(field_times, x, tol=dt.timedelta(days=2))
                          for x in sat_times])
        field_ob = field_ob[ret_t_idx]
        #print t_idx

    innov = [((field_ob[i] - np.mean(field_ob)) - (post_obs[i] - np.mean(post_obs)))**2 for i in xrange(len(post_obs))]
    pos_ubrmse = np.sqrt(np.sum(innov) / len(post_obs))
    innov = [((field_ob[i] - np.mean(field_ob)) - (pri_obs[i] - np.mean(pri_obs)))**2 for i in xrange(len(post_obs))]
    pri_ubrmse = np.sqrt(np.sum(innov) / len(post_obs))

    pos_corrc = np.corrcoef(post_obs, field_ob)[0,1]
    pri_corrc = np.corrcoef(pri_obs, field_ob)[0, 1]

    innov = [(post_obs[i] - field_ob[i])**2 for i in xrange(len(post_obs))]
    pos_rmse = np.sqrt(np.sum(innov) / len(post_obs))
    innov = [(pri_obs[i] - field_ob[i])**2 for i in xrange(len(pri_obs))]
    pri_rmse = np.sqrt(np.sum(innov) / len(pri_obs))

    field_std = np.nanstd(field_ob)
    pri_std = np.nanstd(pri_obs)
    pos_std = np.nanstd(post_obs)
    taylor_pri = (1 + pri_corrc)**4 / (4*(pri_std/field_std + 1/(pri_std/field_std))**2)
    taylor_pos = (1 + pos_corrc)**4 / (4*(pos_std/field_std + 1/(pos_std/field_std))**2)

    pri.close()
    post.close()
    if _all is True:
        ret_val = pri_rmse, pos_rmse, pri_ubrmse, pos_ubrmse, pri_corrc, pos_corrc, taylor_pri, taylor_pos, field_std,\
                  pos_std
    else:
        ret_val = pos_corrc, field_std, pos_std
    return ret_val


def plot_var_paper(var, _dir='ret_code', axes=None, point='508_med', field_obs_on=True):
    post = nc.Dataset(_dir+'/controlvector_post.nc', 'r')
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
    #ax.plot(times, np.array([np.nan]*len(times)), 'X')
    if var == 'sm':
        t_idx = np.array([find_nearest(field_sm['_date'], x)[1] for x in sat_times])
        print t_idx
        field_times = field_sm['_date'][t_idx]
        field_ob = field_sm['sm'][t_idx]
    else:
        field_times = field_laican['_date'][:]
        field_ob = field_laican[var][:]
    palette = sns.color_palette("Greys", 8)
    ax.errorbar(sat_times[post.variables['sim_typ'][:] == 9], post.variables[var][post.variables['sim_typ'][:] == 9],
                #yerr= post.variables[var+'_unc'][post.variables['sim_typ'][:] == 1],
                fmt='o', label='Retrieval output S1', color=palette[2])
    ax.errorbar(sat_times[post.variables['sim_typ'][:] == 34], post.variables[var][post.variables['sim_typ'][:] == 34],
                #yerr=post.variables[var+'_unc'][post.variables['sim_typ'][:] == 2],
                fmt='o', label='Retrieval output S2', color=palette[5])
    if var == 'sm':
        ax.set_ylim([0, 0.5])
    elif var == 'lai':
        ax.set_ylim([0, 8.0])
    if field_obs_on is True:
        ax.plot(field_times, field_ob, '*', label='Field obs', color=palette[7])
    #ax.set_xlabel('Date')
    #ax.set_ylabel(var_dict[var])
    if axes is None:
        fig.autofmt_xdate()
    #plt.legend(frameon=True, fancybox=True, framealpha=0.5)
    return ret_val


def plot_lai(exp_type='flat_hv_b4b5b6b7b8b8a'):
    sns.set_context('poster', font_scale=1.7, rc={'lines.linewidth':1.4, 'lines.markersize':12})
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(20, 20), )
    for x in enumerate(['508', '515', '542']):
        for y in enumerate(['low', 'med', 'high']):
            plot_var_paper('lai', '/export/cloud/nceo/users/if910917/s3_exps/paper/'+x[1]+'_'+y[1]+'/'+exp_type,
                           ax[y[0], x[0]], x[1]+'_'+y[1])
    ax[0, 0].set_title('Field 508')
    ax[0, 1].set_title('Field 515')
    ax[0, 2].set_title('Field 542')
    ax[0, 0].set_ylabel('low \n' + r'Leaf area index (m$^{2}$ m$^{-2}$)')
    ax[1, 0].set_ylabel('med \n' + r'Leaf area index (m$^{2}$ m$^{-2}$)')
    ax[2, 0].set_ylabel('high \n' + r'Leaf area index (m$^{2}$ m$^{-2}$)')
    #ax.set_ylabel('Point')
    #ax.set_title('Field')
    for axi in ax.flat:
        axi.xaxis.set_major_locator(plt.MaxNLocator(4))
        #axi.yaxis.set_major_locator(plt.MaxNLocator(3))
    myFmt = DateFormatter("%Y/%m")
    ax[2,0].xaxis.set_major_formatter(myFmt)
    ax[2, 1].xaxis.set_major_formatter(myFmt)
    ax[2, 2].xaxis.set_major_formatter(myFmt)
    fig.autofmt_xdate()
    return fig


def plot_sm(exp_type='flat_hv_b4b5b6b7b8b8a'):
    sns.set_context('poster', font_scale=1.7, rc={'lines.linewidth':1.4, 'lines.markersize':12})
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(20, 20), )
    for x in enumerate(['508', '515', '542']):
        for y in enumerate(['low', 'med', 'high']):
            plot_var_paper('sm', '/export/cloud/nceo/users/if910917/s3_exps/paper/'+x[1]+'_'+y[1]+'/'+exp_type,
                           ax[y[0],x[0]], x[1]+'_'+y[1])
    ax[0, 0].set_title('Field 508')
    ax[0, 1].set_title('Field 515')
    ax[0, 2].set_title('Field 542')
    ax[0, 0].set_ylabel('low \n'+ r'Soil moisture (m$^{3}$ m$^{-3}$)')
    ax[1, 0].set_ylabel('med \n'+ r'Soil moisture (m$^{3}$ m$^{-3}$)')
    ax[2, 0].set_ylabel('high \n'+ r'Soil moisture (m$^{3}$ m$^{-3}$)')
    #ax.set_ylabel('Point')
    #ax.set_title('Field')
    for axi in ax.flat:
        axi.xaxis.set_major_locator(plt.MaxNLocator(4))
        #axi.yaxis.set_major_locator(plt.MaxNLocator(3))
    myFmt = DateFormatter("%Y/%m")
    ax[2,0].xaxis.set_major_formatter(myFmt)
    ax[2, 1].xaxis.set_major_formatter(myFmt)
    ax[2, 2].xaxis.set_major_formatter(myFmt)
    fig.autofmt_xdate()
    return fig


def plot_s1s2():
    sns.set_context('poster', font_scale=1.7, rc={'lines.linewidth':1.4, 'lines.markersize':12})
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 20), )
    for x in enumerate(['flat_hv_nos2', 'flat_nos1_b4b5b6b7b8b8a', 'flat_hv_b4b5b6b7b8b8a']):
        plot_var_paper('sm', '/export/cloud/nceo/users/if910917/s3_exps/paper/508_med/'+x[1],
                        ax[0,x[0]], '508_med')
        plot_var_paper('lai', '/export/cloud/nceo/users/if910917/s3_exps/paper/508_med/'+x[1],
                        ax[1,x[0]], '508_med')

    ax[0, 0].set_title('S1')
    ax[0, 1].set_title('S2')
    ax[0, 2].set_title('S1 + S2')
    ax[0, 0].set_ylabel(r'Soil moisture (m$^{3}$ m$^{-3}$)')
    ax[1, 0].set_ylabel(r'Leaf area index (m$^{2}$ m$^{-2}$)')
    #ax.set_ylabel('Point')
    #ax.set_title('Field')
    for axi in ax.flat:
        axi.xaxis.set_major_locator(plt.MaxNLocator(4))
        #axi.yaxis.set_major_locator(plt.MaxNLocator(3))
    myFmt = DateFormatter("%Y/%m")
    ax[1,0].xaxis.set_major_formatter(myFmt)
    ax[1, 1].xaxis.set_major_formatter(myFmt)
    ax[1, 2].xaxis.set_major_formatter(myFmt)
    fig.autofmt_xdate()
    return fig



# ------------------------------------------------------------------------------
# Taylor diagram
# ------------------------------------------------------------------------------

def plot_taylor_diagram(set_var):
    sns.set_context('paper', font_scale=1.0, rc={'lines.linewidth':1.4, 'lines.markersize':6})
    #sns.set_style("whitegrid")
    fig = plt.figure()
    palette = sns.color_palette("colorblind", 11)
    #sns.set_style('ticks')
    if set_var == 'sm':
        dia = td.TaylorDiagram(1, fig=fig, rect=111, label='obs.', smin=0.0, smax=2.5)
    elif set_var == 'lai':
        dia = td.TaylorDiagram(1, fig=fig, rect=111, label='obs.', smin=0.0, smax=1.6)
    #for i, (std, corrcoef, name, mark, mss, pal) in enumerate(experimentscvt_a2):
    #    dia.add_sample(std, corrcoef, label=name, marker=mark, ms=mss, mew=1, markerfacecolor=pal,
    #     markeredgecolor='black', markeredgewidth=5, color='white')
    marker = ['v', 's', '^']
    lab = ['S1', 'S2', 'S1+S2']
    color = ['r', 'b', 'g']
    edge = [1.0, 1.5, 2.0]
    for i,x in enumerate(['508_', '515_', '542_']):
        for j,y in enumerate(['low', 'med', 'high']):
            for z in enumerate(['flat_hv_nos2', 'flat_nos1_b4b5b6b7b8b8a', 'flat_hv_b4b5b6b7b8b8a']):
                corr, ob_std, mod_std = save_stats(var=set_var,
                                                   _dir='/export/cloud/nceo/users/if910917/s3_exps/paper/'+x+y+'/'+z[1],
                                                   point=x+y, _all=False)
                dia.add_sample(mod_std/ob_std, corr, label=lab[z[0]]+' '+x+y, marker=marker[z[0]], ms=6,
                               markeredgecolor='black', markeredgewidth=edge[j], color=color[i])  #'white')
    contours = dia.add_contours(levels=8, colors='0.5')
    plt.clabel(contours, inline=1, fontsize=10)
    #fig.legend([dia.samplePoints[0],dia.samplePoints[1],dia.samplePoints[2], dia.samplePoints[3]],
    #           [p.get_label() for p in [dia.samplePoints[0],dia.samplePoints[1],dia.samplePoints[2],
    #                                    dia.samplePoints[3]]],
    #           numpoints=1, loc='upper right')
    # Add a figure legend
    fig.legend(dia.samplePoints,
               [ p.get_label() for p in dia.samplePoints ],
               numpoints=1, prop=dict(size='small'), loc='upper right')
    plt.ylabel('Standard Deviation')
    plt.xlabel('Standard Deviation')
    #plt.ylim((0, 2.5))
    #plt.show()
    return fig


def plot_taylor_diagram_site(set_var, pol='hv', bands='b4b5b6b7b8b8a'):
    sns.set_context('paper', font_scale=1.0, rc={'lines.linewidth':1.4, 'lines.markersize':6})
    sns.set_style("whitegrid")
    fig = plt.figure()
    palette = sns.color_palette("colorblind", 11)
    #sns.set_style('ticks')
    if set_var == 'sm':
        dia = td.TaylorDiagram(1, fig=fig, rect=111, label='obs.', smin=0.0, smax=3.0)
    elif set_var == 'lai':
        dia = td.TaylorDiagram(1, fig=fig, rect=111, label='obs.', smin=0.0, smax=2.0)
    #for i, (std, corrcoef, name, mark, mss, pal) in enumerate(experimentscvt_a2):
    #    dia.add_sample(std, corrcoef, label=name, marker=mark, ms=mss, mew=1, markerfacecolor=pal,
    #     markeredgecolor='black', markeredgewidth=5, color='white')
    marker = ['v', 's', '^']
    lab = ['S1', 'S2', 'S1+S2']
    color = ['r', 'b', 'g']
    for x in ['508_', '515_', '542_']:
        for y in ['low', 'med', 'high']:
            for z in enumerate(['flat_'+pol+'_nos2', 'flat_nos1_'+bands, 'flat_'+pol+'_'+bands]):
                corr, ob_std, mod_std = save_stats(var=set_var,
                                                   _dir='/export/cloud/nceo/users/if910917/s3_exps/paper/'+x+y+'/'+z[1],
                                                   point=x+y, _all=False)
                dia.add_sample(mod_std/ob_std, corr, label=lab[z[0]], marker=marker[z[0]], ms=8,
                               markeredgecolor='black', markeredgewidth=0.8, color='white')
    contours = dia.add_contours(levels=8, colors='0.5')
    plt.clabel(contours, inline=1, fontsize=10)
    fig.legend([dia.samplePoints[0],dia.samplePoints[1],dia.samplePoints[2], dia.samplePoints[3]],
               [p.get_label() for p in [dia.samplePoints[0],dia.samplePoints[1],dia.samplePoints[2],
                                        dia.samplePoints[3]]],
               numpoints=1, loc='upper right')
    # Add a figure legend
    #fig.legend(dia.samplePoints,
    #           [ p.get_label() for p in dia.samplePoints ],
    #           numpoints=1, prop=dict(size='small'), loc='upper right')
    plt.ylabel('Standard Deviation')
    plt.xlabel('Standard Deviation')
    #plt.ylim((0, 2.5))
    #plt.show()
    return fig
