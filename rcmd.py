use_print_future = False
#from __future__ import print_function
#use_print_future = True
import subprocess
import os
import sys
import shutil as sh
import shlex

def run_command(command):
    """
    Function that runs shell command from python and tries to continously monitor the terminal output   
    """
    process = subprocess.Popen(shlex.split(command),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    while True:
        output = process.stdout.readline()
        outerr = process.stderr.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            if use_print_future:
                print(output)
            else:
                # print output.strip()
                if output[-1]=='\n':
                    print output[:-1]
                else:
                    print output
        if outerr:
            print (outerr.strip())
    rc = process.poll()
    return rc


# def run(cmd, ret=False):
#     os.environ['PYTHONUNBUFFERED'] = "1"
#     proc = subprocess.Popen(shlex.split(cmd),
#                             stdout = subprocess.PIPE,
#                             stderr = subprocess.PIPE,
#                             universal_newlines = True,
#                             )
#     stdout = []
#     stderr = []
#     mix = []
#     while proc.poll() is None:
#         line = proc.stdout.readline()
#         if line != "":
#             stdout.append(line)
#             mix.append(line)
#             print(line, end='')
 
#         line = proc.stderr.readline()
#         if line != "":
#             stderr.append(line)
#             mix.append(line)
#             print(line, end='')

#     if ret:
#         return proc.returncode, stdout, stderr, mix
#     else:
#         return proc.returncode
