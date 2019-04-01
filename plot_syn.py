import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import datetime as dt


def plot_state(var, _dir='ret_code', axes=None):
    """
    Function which plots the output of the retrieval tool, including site level observations for comparison.
    :param var: which variable to plot, lai, sm or canht (str)
    :param _dir: directory of retrieval tool output (str)
    :param axes: axes to plot on, if None new axes generated (obj)
    :return: figure and axes of plot (obj)
    """
    post = nc.Dataset(_dir+'/controlvector_post.nc', 'r')
    pri = nc.Dataset(_dir + '/controlvector_prior.nc', 'r')

    if axes is None:
        fig, ax = plt.subplots()
        ret_val = fig, ax
    else:
       ax = axes
       ret_val = ax
    var_dict = {'lai': r'Leaf area index (m$^{2}$ m$^{-2}$)',
                'canht': 'Canopy height (m)',
                'sm': r'Soil moisture (m$^{3}$ m$^{-3}$)'}
    sat_times = nc.num2date(post.variables['time'][:], post.variables['time'].units)

    ax.errorbar(sat_times, pri.variables[var][:],
                yerr=pri.variables[var+'_unc'][:],
                fmt='o', label='prior', color='b', alpha=0.7)
    ax.errorbar(sat_times, post.variables[var][:],
                fmt='x', label='post', color='r')

    post.close()
    pri.close()
    if var == 'sm':
        ax.set_ylim([0, 0.5])
    elif var == 'lai':
        ax.set_ylim([0, 1.0])
    ax.set_xlabel('Date')
    ax.set_ylabel(var_dict[var])
    if axes is None:
        fig.autofmt_xdate()
    plt.legend(frameon=True, fancybox=True, framealpha=0.5)
    return ret_val


def plot_iterate(itfile, var='fct', axes=None):
    """
    Function which plots the output retrieval iteration logfile 'iterate.dat'
    :param itfle: name of log file (str)
    :param axes: axes to plot on, if None new axes generated (obj)
    :return: figure and axes of plot (obj)
    """
    if axes is None:
        fig, ax = plt.subplots()
        ret_val = fig, ax
    else:
       ax = axes
       ret_val = ax

    projg_lst = []
    fct_lst = []
    with open(itfile,'r') as fp:
        do_parse = False
        for line in fp:
            if do_parse:
                tokens = line.split()
                if not len(tokens)==10:
                    break
                else:
                    projg = float(tokens[8].replace('D','E'))
                    projg_lst.append(projg)
                    fct   = float(tokens[9].replace('D','E'))
                    fct_lst.append(fct)
            elif line.find('it   nf  nseg  nact  sub  itls  stepl    tstep     projg        f')>=0:
                do_parse = True
                
    niter = len(fct_lst)
#    print niter
    it_index = np.arange(1,niter+1)
    if var=='fct':
        ax.plot(it_index, fct_lst, label='fct value', color='r', marker='o')
    elif var=='grad':
        ax.plot(it_index, projg_lst, label='gradient', color='b', marker='o')
    ax.set_xlabel('Iteration index')
    ax.set_ylabel('gradient')
    if axes is None:
        fig.autofmt_xdate()
    plt.legend(frameon=True, fancybox=True, framealpha=0.5)
    return ret_val
