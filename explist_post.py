#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
==================================================
PROJECT

Sentinel Synergy Study (S^3)

DESCRIPTION

Postprocessing and analysis of retrieval experients resulting from application of
the prototype retrieval system.

EXAMPLES

available options are listed by invoking 'explist_post.py' on the command line


EXIT STATUS

should be 0 in case of success, 1 otherwise

AUTHOR

The Inversion Lab, Michael Vossbeck <Michael.Vossbeck@Inversion-Lab.com>

==================================================
"""

import sys
import os
from collections import OrderedDict
import re
import fnmatch
import logging
import datetime as dtm
import numpy as np
import numpy.ma as ma
import netCDF4 as nc4
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import pandas as pd


#--------------------------------------------------
#     l o g g i n g
#
FileLogger = logging.getLogger(__file__)
OUT_HDLR = logging.StreamHandler(sys.stdout)
OUT_HDLR.setFormatter(logging.Formatter('%(asctime)s %(levelname)s::%(funcName)s:: %(message)s'))
OUT_HDLR.setLevel( logging.INFO )
FileLogger.addHandler( OUT_HDLR )
FileLogger.setLevel( logging.INFO )


EXPERIMENT_TAGS  = ['prior','reference','complete','nomodel','nos1','nos2']
_MARKER_LST         = ['o',    '^',        '<'   ,    '>'      ,'x',   '+'   ]
_MARKER_LST         = ['.',    '+',        '1',       '2',      '3',   '4'   ]
_MARKER_LST         = ['.',    'x',        '1',       '2',      '3',   '4'   ]
#_MARKER_LST         = ['.', '$pr$', '1', '2', '3', '4']
_COLOR_LST          = ['grey', 'black', 'red', 'blue', 'green', 'magenta']
_COLOR_LST          = ['grey', 'orange', 'red', 'blue', 'green', 'magenta']



def mkdirp_smart(newdir):
    """
    works the way a good mkdir should :)
    - already exists, silently complete
    - regular file in the way, raise an exception
    - parent directory(ies) does not exist, make them as well
    
    @brief create 'newdir' including any non-existent intermediate directories

    @param  string  newdir  relative/absolute pathname to be created,
                            including intermediate directories
    """
    import os

    if os.path.isdir(newdir):
        pass
    elif os.path.exists(newdir):
        msg = "a non-directory path with the same name as the desired " \
              "dir, '{}', already exists.".format(newdir)
        raise RuntimeError(msg)
    else:
        head, tail = os.path.split(newdir)
        if head and not os.path.isdir(head):
            mkdirp_smart(head)
        if tail:
            try:
                os.mkdir(newdir)
            except OSError as exc:
                msg = "newdir={} could not be created on system (exc={})".format(
                    newdir, exc)
                raise RuntimeError(msg)
# ---mkdirp_smart---

def resdirlst_get_tags(resdir_lst):
    tag_lst = []
    for xexp in resdir_lst:
        bexp = os.path.os.path.basename(os.path.normpath(xexp))
        #-- last '_' separated pattern
        tag = bexp.split('_')[-1]
        tag_lst.append(tag)

    return tag_lst

def resdirlst_sort(resdir_lst):

    idx_lst     = []
    list_sorted = []

    for i,tag in enumerate(EXPERIMENT_TAGS):
        for x in resdir_lst:
            if x.find('_'+tag+'_')>0:
                list_sorted.append(x)
                idx_lst.append(i)
                break

    return (list_sorted,idx_lst)
# ---resdirlst_sort---


def ncload_state(ctlvec_file, state, get_unc=False):
    try:
        ncfp = nc4.Dataset(ctlvec_file)
    except IOError:
        msg = "file ***{}*** could not be opened for reading".format(ctlvec_file)
        FileLogger.fatal(msg)
        raise RuntimeError(msg)

    #-- name of state
    data_name = state
    if get_unc:
        data_name = data_name + '_unc'

    data        = ncfp.variables[data_name][:]
    nctime      = ncfp.variables['time']
    timepts     = nc4.num2date(nctime[:], nctime.units)
    if ncfp.variables.has_key('state_typ'):
        try:
            state_types = ncfp.variables['state_typ'][:]
        except KeyError:
            state_types = None
    elif ncfp.variables.has_key('sim_typ'):
        try:
            state_types = ncfp.variables['sim_typ'][:]
        except KeyError:
            state_types = None
    else:
        msg = "neither 'state_typ' nor 'sim_typ' found as dataset in file!"
        raise RuntimeError(msg)

    #-- close handle
    ncfp.close()

    return (data, timepts, state_types)
# ---ncload_state---

def ncload_dataset(data_file, dataset_name):
    try:
        ncfp = nc4.Dataset(data_file)
    except IOError:
        msg = "file ***{}*** could not be opened for reading".format(data_file)
        FileLogger.fatal(msg)
        raise RuntimeError(msg)

    data        = ncfp.variables[dataset_name][:]
    nctime      = ncfp.variables['time']
    timepts     = nc4.num2date(nctime[:], nctime.units)

    #-- close handle
    ncfp.close()

    return (data, timepts)
# ---ncload_dataset---


def ncget_units(ctlvec_file, state):
    units = ''
    try:
        ncfp = nc4.Dataset(ctlvec_file)
    except IOError:
        msg = "file ***{}*** could not be opened for writing"
        FileLogger.fatal(msg)
        raise RuntimeError(msg)
    try:
        units = ncfp.variables[state].getncattr('units')
    except KeyError:
        msg = "units attribute for dataset ---{}--- not available".format(state)
        FileLogger.warn(msg)

    ncfp.close()

    return units

# ---ncget_units---
    

def resdir_load_data(resdir, state, get_prior=False, get_unc=False):
    if get_prior:
        nc_fname = os.path.join(resdir,'controlvector_prior.nc')
    else:
        nc_fname = os.path.join(resdir,'controlvector_post.nc')

    return ncload_state(nc_fname, state, get_unc=get_unc)
# ---resdir_load_data---


def resdir_load_fapar(resdir, get_prior=False, get_unc=False):
    if get_prior:
        nc_fname = os.path.join(resdir,'fapar_prior.nc')
    else:
        nc_fname = os.path.join(resdir,'fapar_post.nc')

    if get_unc:
        dsname = 'fapar_unc'
    else:
        dsname = 'fapar'
    if os.path.exists(nc_fname):
        return ncload_dataset(nc_fname, dsname)
    else:
        return None
# ---resdir_load_data---



def resdirlst_load_data_table(resdir_lst, state, get_unc=False):
    msg = "loading data for ---{}--- from experiments ***{}***".format(
        state, resdir_lst)
    FileLogger.info(msg)

    #-- number of experiments
    nexp = len(resdir_lst)
    
    #-- merge 
    prdata_lst    = []
    prtimepts_lst = []
    prsimtyp_lst  = []
    prfapar_lst   = []
    timepts_merged_set = set()
    for iexp in range(nexp):
        data_, timepts_, st_types = resdir_load_data( resdir_lst[iexp], state,
                                                      get_prior=True, get_unc=get_unc )
        rfpar = resdir_load_fapar(resdir_lst[iexp], get_prior=True, get_unc=get_unc )
        if rfpar is None:
            prfapar_lst.append(None)
        else:
            fpar_data, timepts = rfpar
            prfapar_lst.append(fpar_data) # first in tuple is the data
        prdata_lst.append(data_)
        prtimepts_lst.append(timepts_)
        prsimtyp_lst.append(st_types)
        msg = "iexp={} #timepts={}".format(iexp, timepts_.size)
        FileLogger.info(msg)
        # print "-"*30
        # for x in timepts_:
        #     print x.strftime('%Y%m%dT%H:%M')
        # print "-"*30
        timepts_merged_set = timepts_merged_set.union(set(timepts_))

    #-- 
    timepts_merged = np.array(sorted(list(timepts_merged_set)),dtype=dtm.datetime )
    nt = timepts_merged.size
    msg = "merged time points yields #timepts={}".format(nt)
    FileLogger.info(msg)

    #-- data buffer
    data_fill  = -99999.
    data_table = ma.ones((nexp+1,nt), dtype=data_.dtype)*data_fill
    simtype_a = np.ones((nt,), dtype=np.int32)*-1

    fapar_table = ma.ones((nexp+1,nt), dtype=data_.dtype)*data_fill
    
    #-- merge the prior state
    for it in xrange(nt):
        for iexp in xrange(nexp):
            xtimepts_ = prtimepts_lst[iexp]
            #-- index of 'it' in this experiment
            it_match = np.where(xtimepts_==timepts_merged[it])[0]
            if it_match.size==1:
                iit = it_match[0]
                data_table[0,it] = prdata_lst[iexp][iit]
                #-- ATTENTION:
                #   recent change here: we (likely) have all prior
                #                       with the same time-points,
                #                       for experiments without/or missing observations
                #                       are added as extra target points
                if simtype_a[it]==-1: #not yet set
                    simtype_a[it]    = prsimtyp_lst[iexp][iit]
                elif simtype_a[it]==0 and prsimtyp_lst[iexp][iit]>0:
                    simtype_a[it]    = prsimtyp_lst[iexp][iit]
                elif simtype_a[it]>0 and prsimtyp_lst[iexp][iit]==0:
                    pass
                elif simtype_a[it]!=prsimtyp_lst[iexp][iit]:
                    msg = "detected differing simulation types for same time point {}. ".format(
                        timepts_merged[it].strftime('%Y%m%dT%H:%M:%S'))
                    msg += "existing_entry={} new_entry={}".format(
                        simtype_a[it], prsimtyp_lst[iexp][iit])
                    FileLogger.fatal(msg)
                    raise RuntimeError(msg)
                        
                if prfapar_lst[iexp] is None:
                    pass
                else:
                    if fapar_table[0,it]==data_fill and not ma.is_masked(prfapar_lst[iexp][iit]):
                        fapar_table[0,it] = prfapar_lst[iexp][iit]
                #-- this time point is covered,
                #   assuming all exeriment prior for same date are equal!
                #break

    #-- add posterior values
    for iexp in range(nexp):
        msg = "reading data from directory ***{}***".format(resdir_lst[iexp])
        FileLogger.info(msg)
        data_,xtimepts_, _ = resdir_load_data( resdir_lst[iexp], state,
                                               get_prior=False, get_unc=get_unc )
        rxfpar = resdir_load_fapar(resdir_lst[iexp], get_prior=False, get_unc=get_unc )
        for it in xrange(nt):
            it_match = np.where(xtimepts_==timepts_merged[it])[0]
            if it_match.size==1:
                iit = it_match[0]
                data_table[iexp+1,it] = data_[iit]
                if rxfpar is None:
                    pass
                else:
                    fapar_table[iexp+1,it] = rxfpar[0][iit]
        msg ="reading data from directory ***{}*** DONE".format(resdir_lst[iexp])
        FileLogger.info(msg)
    #-- turn into masked array
    data_table = ma.masked_where(data_table==data_fill, data_table)
    fapar_table = ma.masked_where(fapar_table==data_fill, fapar_table)
    if get_unc:
        fapar_table = ma.masked_where(fapar_table==0., fapar_table)

    return (data_table, fapar_table, timepts_merged, simtype_a)
# ---resdirlst_load_data_table---


def get_legendloc(options):
    legendloc = options.__dict__.get('legendloc', 'best')
    if legendloc=='lr':
        legendloc = 'lower right'
    elif legendloc=='cr':
        legendloc = 'center right'
    return legendloc
# ---get_legendloc---


def dates_get_monlimits(dates_):
    """Function that returns monthly aligned boundary dates for the given list/series of dates

    :param dates_: list/array of datetime objects
    :return: tuple of datetime objects (lower, upper bound)
    :rtype: tuple
    """
    dfirst = dates_[0]
    dlast  = dates_[-1]

    bound_lo = dtm.datetime(dfirst.year, dfirst.month, 1)
    bound_hi = (dtm.datetime(dates_[-1].year, dates_[-1].month, 1)+dtm.timedelta(days=32))
    bound_hi.replace(day=1)
    bound_hi = bound_hi.replace(day=1) - dtm.timedelta(seconds=1)

    return (bound_lo, bound_hi)
# ---dates_get_monlimits---


def ax_add_simtyp_marks(ax, xvalues, statetyp_values, hi=100.):

    ndim1 = statetyp_values.size

    # indices_s1 = np.where( (statetyp_values&1)==1 )
    # print "MVMV::indices_s1:",timepts[indices_s1]
    s1_states = np.ones((ndim1,), dtype=np.float32)*hi
    s1_states = ma.masked_where( (statetyp_values&1)!=1, s1_states )
    s1a_states = np.ones((ndim1,), dtype=np.float32)*hi
    s1a_states = ma.masked_where( (statetyp_values&2**2)!=2**2, s1a_states )
    s1b_states = np.ones((ndim1,), dtype=np.float32)*hi
    s1b_states = ma.masked_where( (statetyp_values&2**3)!=2**3, s1b_states )

    # indices_s2 = np.where( (statetyp_values&2)==2 )
    # print "MVMV::indices_s2:",timepts[indices_s2]
    s2_states = np.ones((ndim1,), dtype=np.float32)*hi
    s2_states = ma.masked_where( (statetyp_values&2)!=2, s2_states )
    s2a_states = np.ones((ndim1,), dtype=np.float32)*hi
    s2a_states = ma.masked_where( (statetyp_values&2**4)!=2**4, s2a_states )
    s2b_states = np.ones((ndim1,), dtype=np.float32)*hi
    s2b_states = ma.masked_where( (statetyp_values&2**5)!=2**5, s2b_states )

    # indices_other = np.where( statetyp_values==0 )
    # print "MVMV::indices_other:",timepts[indices_other]
    so_states = np.ones((ndim1,), dtype=np.float32)*hi
    so_states = ma.masked_where( statetyp_values!=0, so_states)
    mrksiz = 14
    marker = 'v'
    # ax.plot( xvalues, s1_states, linestyle='',
    #          marker=marker, markersize=mrksiz, mfc='red', mec='red', alpha=0.5 )
    # ax.plot( xvalues, s2_states, linestyle='',
    #          marker=marker, markersize=mrksiz, mfc='blue', mec='blue', alpha=0.5 )
    ax.plot( xvalues, s1a_states, linestyle='',
             marker=marker, markersize=mrksiz, mfc='red', mec='red', alpha=0.8 )
    ax.plot( xvalues, s1b_states, linestyle='',
             marker=marker, markersize=mrksiz, mfc='red', mec='red', alpha=0.5 )
    ax.plot( xvalues, s2a_states, linestyle='',
             marker=marker, markersize=mrksiz, mfc='blue', mec='blue', alpha=0.8 )
    ax.plot( xvalues, s2b_states, linestyle='',
             marker=marker, markersize=mrksiz, mfc='blue', mec='blue', alpha=0.5 )
    ax.plot( xvalues, so_states, linestyle='',
             marker=marker, markersize=mrksiz, mfc='green', mec='green', alpha=0.5 )
# ---ax_add_simtyp_marks---


def explist_unctable(options):
    resdir_lst = options.resultdirs
    state      = options.state
    outdir     = options.outdir

    #-- number of experiments
    nexp = len(resdir_lst)

    #-- determine experiment tags
    exptag_lst = resdirlst_get_tags(resdir_lst)
    msg = "detected #={} experiments, tags: ***{}***".format(len(exptag_lst),exptag_lst)
    FileLogger.info(msg)

    #-- maximal tag length
    tag_lenghts = np.array([len(tag) for tag in exptag_lst], dtype=np.int32)
    tagmxlen = tag_lenghts.max()
    tl = str(tagmxlen+3)
    msg = "load data from result directories..."
    FileLogger.info(msg)
    unc_table, fparunc_table, timepts, timepts_typ = resdirlst_load_data_table(resdir_lst, state, get_unc=True)
    msg = "...loading data is done."
    FileLogger.info(msg)
    #-- #timepoints
    nt = timepts.size

    #-- output name
    act_outname = "{}_unc_comparison".format(state)
    for tag in exptag_lst:
        act_outname += "_" + tag
    act_outname += ".txt"
    if outdir!=None:
        act_outname = os.path.join(outdir, act_outname)
        mkdirp_smart(os.path.dirname(act_outname))

    with open(act_outname, 'w') as fp:
        header = ("{:16s}{:>"+tl+"s}").format('date','prior')
        for tag in exptag_lst:
            header += ("{:>"+tl+"s}").format(tag)
        fp.write(header+'\n')
        for i in range(unc_table.shape[1]):
            fp.write('{}'.format(timepts[i].strftime('%Y-%m-%dT%H:%M')))
            for j in range(nexp+1):
                if ma.is_masked(unc_table[j,i]):
                    fp.write(("{:>"+tl+"s}").format('--'))
                else:
                    fp.write(("{:"+tl+".4f}").format(unc_table[j,i]))
            fp.write('\n')
        msg = "sucessfully created file ***{}***".format(act_outname)
        FileLogger.info(msg)

    #-- uncertainty reduction
    unc_prior = unc_table[0,:]
    uncred_table = ma.empty((nexp,nt), dtype=unc_table.dtype) #allocate table
    for iexp in range(nexp):
        uncred_table[iexp,:] = 1. - unc_table[iexp+1,:]/unc_prior
    #-- convert to percent
    uncred_table *= 100

    #-- output name
    act_outname = "{}_uncred_comparison".format(state)
    for tag in exptag_lst:
        act_outname += "_" + tag
    act_outname += ".txt"
    if outdir!=None:
        act_outname = os.path.join(outdir, act_outname)
        mkdirp_smart(os.path.dirname(act_outname))

    with open(act_outname, 'w') as fp:
        header = ("{:16s}").format('date')
        for tag in exptag_lst:
            header += ("{:>"+tl+"s}").format(tag)
        fp.write(header+'\n')
        for i in range(uncred_table.shape[1]):
            fp.write('{}'.format(timepts[i].strftime('%Y-%m-%dT%H:%M')))
            for j in range(uncred_table.shape[0]):
                if ma.is_masked(uncred_table[j,i]):
                    fp.write(("{:>"+tl+"s}").format('--'))
                else:
                    fp.write(("{:"+tl+".4f}").format(uncred_table[j,i]))
            fp.write('\n')
        msg = "sucessfully created file ***{}***".format(act_outname)
        FileLogger.info(msg)

# ---explist_unctable---


def explist_uncred_lineplot(options):

    resdir_lst = options.resultdirs
    state      = options.state
    plt_fpar   = options.fapar

    #-- access plot settings
    markersize     = options.__dict__.get('markersize', 8)
    xticklabsize   = options.__dict__.get('xticklabsize', 8)
    yticklabsize   = options.__dict__.get('yticklabsize', 8)

    #-- determine experiment tags
    exptag_lst = resdirlst_get_tags(resdir_lst)
    nexp = len(exptag_lst)
    msg = "detected #={} experiments, tags: ***{}***".format(nexp,exptag_lst)
    FileLogger.info(msg)

    #-- maximal tag length
    tag_lenghts = np.array([len(tag) for tag in exptag_lst], dtype=np.int32)
    tagmxlen = tag_lenghts.max()
    tl = str(tagmxlen+3)
    msg = "load data from result directories..."
    FileLogger.info(msg)
    stateunc_table, fparunc_table, timepts, timepts_typ = resdirlst_load_data_table(resdir_lst, state, get_unc=True)
    msg = "...loading data is done."
    FileLogger.info(msg)
    #-- #timepoints
    nt = timepts.size
    if plt_fpar:
        unc_table = fparunc_table
    else:
        unc_table = stateunc_table

    # print 'MVMV::unc_table='
    # print '='*40
    # print unc_table
    # print '='*40
    #-- uncertainty reduction
    unc_prior = unc_table[0,:]
    uncred_table = ma.empty((nexp,nt), dtype=unc_table.dtype) #allocate table
    for iexp in range(nexp):
        uncred_table[iexp,:] = 1. - unc_table[iexp+1,:]/unc_prior
    #-- convert to percent
    uncred_table *= 100

    #-- axis values and limits
    xvalues = timepts
    if options.trange==None:
        xlim_lo, xlim_hi = dates_get_monlimits(xvalues)
    else:
        xlim_lo = dtm.datetime.strptime(options.trange[0], '%Y%m%d' )
        xlim_hi = dtm.datetime.strptime(options.trange[1], '%Y%m%d' )

    ylim_lo = -2. #-- set negative value to avoid the zero-valued reference case to
                  #   appear directly on the axis
    ylim_hi = 100. + 2.
    if options.yhi!=None:
        ylim_hi = options.yhi
    if options.ylo!=None:
        ylim_lo = options.ylo

    msg = "xlim_lo={} xlim_hi={} ylim_lo={} ylim_hi={}".format(xlim_lo, xlim_hi, ylim_lo, ylim_hi)
    FileLogger.info(msg)

    #-- create figure
    fig = plt.figure(dpi=options.dpi)
    ax = fig.add_subplot(1,1,1)

    for iexp in range(nexp):
        marker = _MARKER_LST[iexp+1]
        color  = _COLOR_LST[iexp+1]
        label  = exptag_lst[iexp]
        if iexp==0:
            markersize = 10
        ax.plot( xvalues, uncred_table[iexp,:], linestyle='',
                 marker=marker, markersize=markersize, mfc=color, mec=color, label=label )

    #-- add state-type marks
    ax_add_simtyp_marks(ax, xvalues, timepts_typ, hi=ylim_hi)

    #-- xaxis labeling
    mpl.pyplot.setp(ax.get_xaxis().get_majorticklabels(), fontsize=xticklabsize)
    mpl.pyplot.setp(ax.get_yaxis().get_majorticklabels(), fontsize=yticklabsize)
    if xlim_lo==dtm.datetime(2017,5,1) and xlim_hi==dtm.datetime(2017,7,31,23,59,59):
        locator   = mpl.dates.DayLocator(bymonthday=[1,5,10,15,20,25,30])
        formatter = mpl.dates.DateFormatter('%b %d')
        mondays = mpl.dates.WeekdayLocator(mpl.dates.MONDAY) # major ticks on the mondays
	alldays = mpl.dates.DayLocator(interval=1)           # minor ticks on the days
	weekFormatter = mpl.dates.DateFormatter('%b %d')     # e.g., Jan 12
	dayFormatter = mpl.dates.DateFormatter('%d')
        ax.xaxis.set_major_locator(mondays)
        ax.xaxis.set_major_formatter(weekFormatter)
        ax.xaxis.set_minor_locator(alldays)
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    #-- finalise plot
#    plt.tight_layout()
    ax.set_xlim(xlim_lo, xlim_hi)
    ax.set_ylim(ylim_lo, ylim_hi)
    #-- axis labels
    # ylabel = r"$\frac{\sigma(LAI_{prior})-\sigma(LAI_{post})}{\sigma(LAI_{prior})}$"
    # ylabel += '\n' + ' [percent]'
    ylabel = 'reduction'  '[%]'
    ylabel = "[%]"
    ax.set_ylabel(ylabel)
    #-- grid background
    # Don't allow the axis to be on top of your data
    ax.set_axisbelow(True)
    # Customize the major grid
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    # Customize the minor grid
    ax.grid(which='minor', axis='x', linestyle=':', linewidth='0.25', color='black')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend( handles[:], labels[:],
               loc=get_legendloc(options), prop={'size':6} )

    #-- title
    if plt_fpar:
        data_tag = 'fapar'
    else:
        data_tag = state
    if options.title!=None:
        title = options.title
    else:
        title = "Uncertainty reduction of {}".format(data_tag.upper())
    fig.suptitle(title)

    #-- filename
    if options.outname!=None:
        outname  = options.outname
    else:
        outname = "{}_uncred_{}-{}".format(data_tag, xlim_lo.strftime('%Y%m%d'), xlim_hi.strftime('%Y%m%d'))
        for tag in exptag_lst:
            outname += "_" + tag
        outname += ".png"
    if not os.path.isabs(outname) and options.outdir!=None:
        outname = os.path.join(options.outdir, outname)
    mkdirp_smart(os.path.dirname(outname))

    #-- save figure
    plt.savefig( outname, dpi=options.dpi)
    plt.close(fig)

    msg = "graphics file ***{}*** has been written".format(outname)
    FileLogger.info(msg)

# ---explist_uncred_lineplot---


def uncred_single_lineplot(options):
    ctlvec_prfile = options.ctlvec_prior_file
    ctlvec_pofile = options.ctlvec_post_file
    state       = options.state

    #-- access plot settings
    markersize = options.__dict__.get('markersize', 8)
    xticklabsize   = options.__dict__.get('xticklabsize', 8)
    yticklabsize   = options.__dict__.get('yticklabsize', 10)

    #-- load data
    msg = "load data from file ***{}***".format(ctlvec_file)
    FileLogger.info(msg)
    pr_stateunc, timepts, timepts_typ = ncload_state( ctlvec_prior_file, state, get_unc=True )
    po_stateunc, _, _                 = ncload_state( ctlvec_post_file, state, get_unc=True )
    msg = "...loading data is done."
    FileLogger.info(msg)

    #-- number of experiments
    nt = timepts.size

    #-- uncertainty reduction
    ur_state = ma.empty((nt,), dtype=pr_stateunc.dtype)
    ur_state = (1. - po_stateunc/pr_stateunc)
    ur_state *= 100. #-- convert to percent

    #-- axis values and limits
    xvalues = timepts
    if options.trange==None:
        xlim_lo, xlim_hi = dates_get_monlimits(xvalues)
    else:
        xlim_lo = dtm.datetime.strptime(options.trange[0], '%Y%m%d' )
        xlim_hi = dtm.datetime.strptime(options.trange[1], '%Y%m%d' )

    ylim_lo = 0.
    ylim_hi = 100.
    msg = "xlim_lo={} xlim_hi={} ylim_lo={} ylim_hi={}".format(xlim_lo, xlim_hi, ylim_lo, ylim_hi)
    FileLogger.info(msg)

    #-- create figure
    fig = plt.figure(dpi=options.dpi)
    ax = fig.add_subplot(1,1,1)

    marker = _MARKER_LST[0]
    color  = _COLOR_LST[0]
    label  = EXPERIMENT_TAGS[0]
    ax.plot( xvalues, ur_state, linestyle='',
             marker=marker, markersize=5, mfc=color, mec=color )

    #-- add state-typ marks
    ax_add_simtyp_marks(ax, xvalues, timepts_typ, hi=ylim_hi)

    #-- xaxis labeling
    mpl.pyplot.setp(ax.get_xaxis().get_majorticklabels(), fontsize=xticklabsize)
    mpl.pyplot.setp(ax.get_yaxis().get_majorticklabels(), fontsize=yticklabsize)
    if xlim_lo.year==xlim_hi.year and xlim_lo.month==xlim_hi.month:
        locator = mpl.dates.DayLocator(bymonthday=[1,5,10,15,20,25,30])
        formatter = mpl.dates.DateFormatter('%b %d')
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

    #-- finalise plot
    ax.set_xlim(xlim_lo, xlim_hi)
    ax.set_ylim(ylim_lo, ylim_hi)
    ax.set_ylabel('[percent]')
    ax.grid(axis='both')

    #-- title
    if options.title!=None:
        title = options.title
    else:
        title = "Uncertainty reduction of {}".format(state.upper())
    fig.suptitle(title)

    #-- filename
    if options.outname!=None:
        outname  = options.outname
    else:
        outname = '{}_uncred.png'.format(state)
    if not os.path.isabs(outname) and options.outdir!=None:
        outname = os.path.join(options.outdir, outname)
    mkdirp_smart(os.path.dirname(outname))

    #-- save figure
    plt.savefig( outname, dpi=options.dpi)
    plt.close(fig)

    msg = "graphics file ***{}*** has been written".format(outname)
    FileLogger.info(msg)

# ---uncred_single_lineplot---


def plot_target(options):
    datafile     = options.targetfile
    dataset_name = options.name

    #-- access plot settings
    markersize = options.__dict__.get('markersize', 8)
    xticklabsize   = options.__dict__.get('xticklabsize', 8)
    yticklabsize   = options.__dict__.get('yticklabsize', 10)

    #-- load data
    msg = "load data from file ***{}***".format(datafile)
    FileLogger.info(msg)
    data, timepts = ncload_dataset( datafile, dataset_name )
    msg = "...loading data is done."
    FileLogger.info(msg)

    #-- number of experiments
    nt = timepts.size

    #-- uncertainty reduction
    #-- axis values and limits
    xvalues = timepts
    if options.trange==None:
        xlim_lo, xlim_hi = dates_get_monlimits(xvalues)
    else:
        xlim_lo = dtm.datetime.strptime(options.trange[0], '%Y%m%d' )
        xlim_hi = dtm.datetime.strptime(options.trange[1], '%Y%m%d' )

    ylim_lo = 0.
    ylim_hi = data.max()
    msg = "xlim_lo={} xlim_hi={} ylim_lo={} ylim_hi={}".format(xlim_lo, xlim_hi, ylim_lo, ylim_hi)
    FileLogger.info(msg)

    #-- create figure
    fig = plt.figure(dpi=options.dpi)
    ax = fig.add_subplot(1,1,1)

    marker = 'o'
    color  = 'green'
    ax.plot( xvalues, data, linestyle='',
             marker=marker, markersize=5, mfc=color, mec=color )

    #-- add state-typ marks
    # ax_add_simtyp_marks(ax, xvalues, timepts_typ, hi=ylim_hi)

    #-- xaxis labeling
    mpl.pyplot.setp(ax.get_xaxis().get_majorticklabels(), fontsize=xticklabsize)
    mpl.pyplot.setp(ax.get_yaxis().get_majorticklabels(), fontsize=yticklabsize)
    if xlim_lo.year==xlim_hi.year and xlim_lo.month==xlim_hi.month:
        locator = mpl.dates.DayLocator(bymonthday=[1,5,10,15,20,25,30])
        formatter = mpl.dates.DateFormatter('%b %d')
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

    #-- finalise plot
    ax.set_xlim(xlim_lo, xlim_hi)
    ax.set_ylim(ylim_lo, ylim_hi)
    ax.set_ylabel('')
    ax.grid(axis='both')

    #-- title
    if options.title!=None:
        title = options.title
    else:
        title = "Time-series of {}".format(dataset_name.upper())
    fig.suptitle(title)

    #-- filename
    if options.outname!=None:
        outname  = options.outname
    else:
        outname = '{}.png'.format(dataset_name)
    if not os.path.isabs(outname) and options.outdir!=None:
        outname = os.path.join(options.outdir, outname)
    mkdirp_smart(os.path.dirname(outname))

    #-- save figure
    plt.savefig( outname, dpi=options.dpi)
    plt.close(fig)

    msg = "graphics file ***{}*** has been written".format(outname)
    FileLogger.info(msg)

# ---plot_target---


def explist_state_lineplot(options):

    resdir_lst = options.resultdirs
    state      = options.state
    show_prior = options.show_prior
    plt_fpar   = options.fapar

    #-- access plot settings
    markersize = options.__dict__.get('markersize', 8)
    xticklabsize   = options.__dict__.get('xticklabsize', 8)
    yticklabsize   = options.__dict__.get('yticklabsize', 10)


    #-- determine experiment tags
    exptag_lst = resdirlst_get_tags(resdir_lst)
    nexp = len(exptag_lst)
    msg = "detected #={} experiments, tags: ***{}***".format(nexp,exptag_lst)
    FileLogger.info(msg)

    #-- maximal tag length
    tag_lenghts = np.array([len(tag) for tag in exptag_lst], dtype=np.int32)
    tagmxlen = tag_lenghts.max()
    tl = str(tagmxlen+3)
    msg = "load data from result directories..."
    FileLogger.info(msg)
    state_table, fpar_table, timepts, timepts_typ = resdirlst_load_data_table(resdir_lst, state, get_unc=False)
    stateunc_table, fparunc_table, _, _ = resdirlst_load_data_table(resdir_lst, state, get_unc=True)
    msg = "...loading data is done."
    FileLogger.info(msg)
    #-- #timepoints
    nt = timepts.size

    if plt_fpar:
        unc_table = fparunc_table
        data_table = fpar_table
    else:
        unc_table = stateunc_table
        data_table = state_table

    #-- get units (read from prior, reference case)
    nc_file_new = os.path.join(resdir_lst[0],'controlvector_prior.nc')
    if plt_fpar:
        units = ''
    else:
        units = ncget_units(nc_file_new, state)
        
    #-- number of experiments
    nexp = len(resdir_lst)
    nt = timepts.size

    if plt_fpar:
        data_tag = 'fapar'
    else:
        data_tag = state


    #-- axis values and limits
    xvalues = timepts
    if options.trange==None:
        xlim_lo, xlim_hi = dates_get_monlimits(xvalues)
    else:
        xlim_lo = dtm.datetime.strptime(options.trange[0], '%Y%m%d' )
        xlim_hi = dtm.datetime.strptime(options.trange[1], '%Y%m%d' )
    ylim_lo = 0.
    ylim_hi = data_table.max()
    if options.yhi!=None:
        ylim_hi = options.yhi
    if options.ylo!=None:
        ylim_lo = options.ylo
    elif data_tag=='lai':
        ylim_hi = 1.1
    elif data_tag=='canht':
        ylim_hi = 1.5
    elif data_tag=='sm':
        ylim_hi = 0.5
    elif data_tag=='fapar':
        ylim_hi = 1.

    if data_table.max()>ylim_hi:
        msg = "selected ylim_hi={} is below datamax={}".format(ylim_hi, data_table.max())
        FileLogger.warn(msg)
    msg = "xlim_lo={} xlim_hi={} ylim_lo={} ylim_hi={}".format(xlim_lo, xlim_hi, ylim_lo, ylim_hi)
    FileLogger.info(msg)

    #-- create figure
    fig = plt.figure(dpi=options.dpi)
    ax = fig.add_subplot(1,1,1)

    #-- plot prior
    if show_prior:
        plt_data = data_table[0,:]
        marker = _MARKER_LST[0]
        color  = _COLOR_LST[0]
        label  = 'prior'
        ax.plot( xvalues, plt_data, linestyle='',
                 marker=marker, markersize=markersize, mfc=color, mec=color, label=label )

    for iexp in range(nexp):
        plt_data = data_table[iexp+1,:] #-- ATTENTION:prior is at position 0 !
        marker = _MARKER_LST[iexp+1]
        color  = _COLOR_LST[iexp+1]
        label  = exptag_lst[iexp]
        ax.plot( xvalues, plt_data, linestyle='',
                 marker=marker, markersize=markersize, mfc=color, mec=color, label=label )

    #-- add state-typ marks
    ax_add_simtyp_marks(ax, xvalues, timepts_typ, hi=ylim_hi)

    #-- xaxis labeling
    mpl.pyplot.setp(ax.get_xaxis().get_majorticklabels(), fontsize=xticklabsize)
    mpl.pyplot.setp(ax.get_yaxis().get_majorticklabels(), fontsize=yticklabsize)
    if xlim_lo==dtm.datetime(2017,5,1) and xlim_hi==dtm.datetime(2017,7,31,23,59,59):
        locator   = mpl.dates.DayLocator(bymonthday=[1,5,10,15,20,25,30])
        formatter = mpl.dates.DateFormatter('%b %d')
        mondays = mpl.dates.WeekdayLocator(mpl.dates.MONDAY) # major ticks on the mondays
	alldays = mpl.dates.DayLocator(interval=1)           # minor ticks on the days
	weekFormatter = mpl.dates.DateFormatter('%b %d')     # e.g., Jan 12
	dayFormatter = mpl.dates.DateFormatter('%d')
        ax.xaxis.set_major_locator(mondays)
        ax.xaxis.set_major_formatter(weekFormatter)
        ax.xaxis.set_minor_locator(alldays)
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    #-- finalise plot
    ax.set_xlim(xlim_lo, xlim_hi)
    ax.set_ylim(ylim_lo, ylim_hi)
#    ax.set_xlabel('time')
    ax.set_ylabel(data_tag.upper() + '\n' + '['+units+']')
    ax.grid(axis='both')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend( handles, labels,
               loc=get_legendloc(options), prop={'size':6} )

    if options.title!=None:
        title = options.title
    else:
        if data_tag=='fapar':
            title = "Target variable: FAPAR"
        else:
            title = "statevector component {}".format(data_tag.upper())
    fig.suptitle(title)

    #-- filename
    if options.outname!=None:
        outname  = options.outname
    else:
        outname = "{}_{}-{}".format(data_tag, xlim_lo.strftime('%Y%m%d'), xlim_hi.strftime('%Y%m%d'))
        for tag in exptag_lst:
            outname += "_" + tag
        outname += ".png"
    if not os.path.isabs(outname) and options.outdir!=None:
        outname = os.path.join(options.outdir, outname)
    mkdirp_smart(os.path.dirname(outname))

    #-- save figure
    plt.savefig( outname, dpi=options.dpi)
    plt.close(fig)

    msg = "graphics file ***{}*** has been written".format(outname)
    FileLogger.info(msg)

# ---explist_state_lineplot---


#==================================================
#
#               create_argument_parser
#
#
def create_argument_parser(progname=None):
    import argparse

    #%%%%%%%%%%%%%%%%%%%
    #
    #               AliasedSubParsersAction
    #
    # Purpose:
    # - provide aliases for 'sub commands'
    #
    # NOTE:
    # - this feature is *not* available with python 2.7.6
    #   (it will be included (at least) since python 3.3.x)
    #
    #
    class AliasedSubParsersAction(argparse._SubParsersAction):
        class _AliasedPseudoAction(argparse.Action):
            def __init__(self, name, aliases, help):
                dest = name
                if aliases:
                    dest += ' (%s)' % ','.join(aliases)
                sup = super(AliasedSubParsersAction._AliasedPseudoAction, self)
                sup.__init__(option_strings=[], dest=dest, help=help) 

        def add_parser(self, name, **kwargs):
            if 'aliases' in kwargs:
                aliases = kwargs['aliases']
                del kwargs['aliases']
            else:
                aliases = []

            parser = super(AliasedSubParsersAction, self).add_parser(name, **kwargs)

            # Make the aliases work.
            for alias in aliases:
                self._name_parser_map[alias] = parser
            # Make the help text reflect them, first removing old help entry.
            if 'help' in kwargs:
                help = kwargs.pop('help')
                self._choices_actions.pop()
                pseudo_action = self._AliasedPseudoAction(name, aliases, help)
                self._choices_actions.append(pseudo_action)

            return parser
    # ---AliasSubParsersAction

    def _add_plot_options(aparser):
        aparser.add_argument( '--outname',
                              help="""write graphics to this file""" )
        aparser.add_argument( '--outdir',
                              help="""where to place generated file(s).""" )
        aparser.add_argument( '--title',
                              help="""user-defined title (may be ignored/modified in case the plot command would yield multiple plots.""" )
        aparser.add_argument( '--dpi',
                              type=int,
                              default=150,
                              help="""dots-per-inch (default:%(default)s)""" )
        aparser.add_argument( '--legendloc',
                              default='best',
                              choices=['best','center','right','left',
                                       'lr','cr'],
                              help="""positioning of legend (default:%(default)s)""" )
        aparser.add_argument( '--trange',
                              nargs=2,
                              help="""restrict plotting to this temporal range yyyymmdd YYYYMMDD""" )
        aparser.add_argument( '--yhi',
                              type=float,
                              help="""user specified maximum value at y-axis""" )
        aparser.add_argument( '--ylo',
                              type=float,
                              help="""user specified minimum value at y-axis""" )
    # ---_add_plot_options---

    parser = argparse.ArgumentParser( prog=progname, usage=globals()['__doc__'] )

    #-------------------
    #     c o m m o n   o p t i o n s
    #
    parser.add_argument( '--verbose','-v',
                         action='store_true',
                         help="""dump some more debugging messages to terminal.""" )

    #----------------------------
    #     s u b c o m m a n d s
    #
    subparsers = parser.add_subparsers( title='Available Subcommands',
                                        metavar='CMDS',
                                        description='',
                                        dest='subcmds',
                                        help=''
                                        )

    #-------------------
    #--   explist_unctable
    xparser = subparsers.add_parser( 'explist_unctable',
                                     help="""creating uncertainty table for list of experiments""" )
    xparser.add_argument( 'resultdirs',
                          nargs='+',
                          help="""result directories resulting from retrieval system experiments""" )
    xparser.add_argument( '--state',
                          default='lai',
                          choices=['lai','canht','sm'],
                          help="""which state compoenent to plot (default:%(default)s)""" )
    xparser.add_argument( '--fapar',
                          action='store_true',
                          help="""whether to plot FAPAR instead of a state""" )
    xparser.add_argument( '--outdir',
                          help="""output directory selected by user""" )
    #-------------------
    #--   explist_uncred_lineplot
    xparser = subparsers.add_parser( 'explist_uncred_lineplot',
                                     help="""creating single plot for the uncertainty reduction of single state variable and all experiments""" )
    xparser.add_argument( 'resultdirs',
                          nargs='+',
                          help="""result directories resulting from retrieval system experiments""" )
    xparser.add_argument( '--state',
                          default='lai',
                          choices=['lai','canht','sm'],
                          help="""which state compoenent to plot (default:%(default)s)""" )
    xparser.add_argument( '--fapar',
                          action='store_true',
                          help="""whether to plot FAPAR instead of a state""" )
    _add_plot_options(xparser)

    #-------------------
    #--   explist_state_lineplot
    xparser = subparsers.add_parser( 'explist_state_lineplot',
                                     help="""creating single plot for a songle state variable and all experiments""" )
    xparser.add_argument( 'resultdirs',
                          nargs='+',
                          help="""result directories resulting from retrieval system experiments""" )
    xparser.add_argument( '--state',
                          default='lai',
                          choices=['lai','canht','sm','lai_unc','canht_unc','sm_unc'],
                          help="""which state compoenent to plot (default:%(default)s)""" )
    xparser.add_argument( '--fapar',
                          action='store_true',
                          help="""whether to plot FAPAR instead of a state""" )
    xparser.add_argument( '--show_prior',
                          action='store_true',
                          help="""whether to show the prior state on the plot as well.""" )
    _add_plot_options(xparser)


    #--   uncred_single
    xparser = subparsers.add_parser( 'uncred_single',
                                     help="""creating line plot of the uncertainty reduction from result file""" )
    xparser.add_argument( 'ctlvec_prior_file',
                          help="""prior control vector file as generated by the prototype retrieval system""" )
    xparser.add_argument( 'ctlvec_post_file',
                          help="""posterior control vector file as generated by the prototype retrieval system""" )
    xparser.add_argument( '--state',
                          default='lai',
                          choices=['lai','canht','sm'],
                          help="""which state compoenent to plot (default:%(default)s)""" )
    _add_plot_options(xparser)

    xparser = subparsers.add_parser( 'plot_target',
                                     help="""plot time-series of target variable""" )
    xparser.add_argument( 'targetfile',
                          help="""name of NetCDF target file""" )
    xparser.add_argument( '--name',
                          default='fapar',
                          help="""name of dataset in file""" )
    _add_plot_options(xparser)

    #-- return main parser
    return parser

# ---create_argument_parser---


def main(options):

    if options.subcmds=='explist_unctable':
        explist_unctable(options)

    if options.subcmds=='explist_uncred_lineplot':
        explist_uncred_lineplot(options)

    if options.subcmds=='explist_state_lineplot':
        explist_state_lineplot(options)

    if options.subcmds=='uncred_single':
        uncred_single_lineplot(options)

    if options.subcmds=='plot_target':
        plot_target(options)
# ---main---


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
#                    M A I N
#
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == '__main__':

    progname = os.path.basename(__file__) #determine filename

    #-----------------------------
    #     P R O G R A M   S T A R T
    #
    fmt = "%Y-%m-%dT%H:%M:%S.%f"
    ttstart = dtm.datetime.now()
    FileLogger.info("{}::PROGRAM START::{}".format(progname, ttstart.strftime(fmt)))
    FileLogger.info( "command-line: {}".format(' '.join(sys.argv)) )

    #--   p a r s e   c o m m a n d   l i n e
    parser = create_argument_parser(progname=progname)
    options = parser.parse_args()

    #debug:
    if options.verbose:
        OUT_HDLR.setLevel( logging.DEBUG )
        FileLogger.setLevel( logging.DEBUG )

    main(options)

# * * * M A I N * * *
