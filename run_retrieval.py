import numpy as np
import subprocess
import os
import sys
import shutil as sh


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


def run_retrieval_tool(out_dir, site='508_med', date_start='20170323', date_end='20170720', obs_s1=True,
                       obs_s2=True, prior=False, run_ret=True,):
    """
    Funtion that runs the Sentinel Synergy Retrieval Tool for desired date range
    :param out_dir: Directory where to store retrieval output (str)
    :param site: which site to run retrieval tool at (str)
    :param date_start: start date for retrievals (str)
    :param date_end: end date for retrievals (str)
    :param obs_s1: use Sentinel 1 observations in retrieval (bool)
    :param obs_s2: use Sentinel 2 observations in retrieval (bool)
    :param prior: use a JULES prior estimate in retrievals (bool)
    :param run_ret: run retrieval (bool)
    :return:
    """
    if obs_s1 is True:
        obs_s1 = '../../data/s1_obs/mni_s1_' + site + '_hv.csv'
    if obs_s2 is True:
        obs_s2 = '../../data/s2_obs/mni_s2_' + site + '_b4b5b6b7b8b8a.csv'
    if prior is True:
        prior = '../../data/field_obs/retr_jules_prior.csv'
    dir_path = os.path.dirname(os.path.realpath(__file__))
    #print dir_path
    ret_tool_dic = {'time_start': date_start, 'time_end': date_end, 'site_nml': dir_path+'/site.nml',
                    'states_file': prior, 'obs_s1': obs_s1, 'obs_s2': obs_s2, 'no_use_prior': False, 'gtol': 1e0,
                    #'no_use_states': False,
                    'dynmodunc_inifile': dir_path+'/data/dynmod/dynmod.ini', 's1_unc': 1.6, 's2_relunc': 0.05,
                    's2_uncfloor': 0.02, 's1_vv_uncfloor': 0.004, 'ctlvec_relunc': [0.01, 0.5, 0.05, 0.5],
                    'ctlvec_uncfloor': [0.001, 3.0, 0.1, 0.05], 's1_vh_uncfloor':0.00001}
    if run_ret is True:
        if os.path.isdir(out_dir) is True:
            sh.rmtree(out_dir)
        subprocess.call(['cp', '-R', dir_path+'/retrieval_tool_examples/ret_code1.7', out_dir])
        os.chdir(out_dir)
        subprocess.call(['make', 'clean'])
        subprocess.call(['make', 'setup'])
        cmd = ['bin/rs_pre.py', 'pre_general']
        for key in ret_tool_dic.keys():
            if ret_tool_dic[key] is not False:
                if type(ret_tool_dic[key]) is str:
                    #print 'str, '+key
                    cmd.append('--'+key)
                    cmd.append(ret_tool_dic[key])
                elif type(ret_tool_dic[key]) is float:
                    #print 'float, '+key
                    cmd.append('--'+key)
                    cmd.append(str(ret_tool_dic[key]))
                elif type(ret_tool_dic[key]) is list:
                    cmd.append('--'+key)
                    for x in ret_tool_dic[key]:
                        #print x
                        cmd.append(str(x))
                else:
                    cmd.append('--'+key)
        subprocess.call(cmd)
        subprocess.call(['make', 'retrieval' , 'RETRARGXTRA=--no_targets'])
        # subprocess.call(['make', 'mba'])
        os.chdir(dir_path)
    return ret_tool_dic