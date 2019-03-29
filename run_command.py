import subprocess
import os
import sys
import shutil as sh
import shlex

def run_command(command):
    """
    Function that runs shell command from python and tries to continously monitor the terminal output   
    """
    process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print output.strip()
    rc = process.poll()
    return rc
