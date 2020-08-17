#!/bin/python

from subprocess import Popen, PIPE
from os import listdir
from os.path import isfile, join

from config import *

def read():
    samples = [join(IN_DIR, f) for f in listdir(IN_DIR) if isfile(join(IN_DIR, f))]

    return samples


def runCMD(cmd, outputfile=None):
    stdoutHandle = PIPE
    if outputfile != None:
        stdoutHandle = outputfile

    p = Popen(cmd, stdout=stdoutHandle, stderr=PIPE)
    stdout, stderr = p.communicate()

    if "ERROR" in stderr:
        raise Exception(stderr)

    return stdout
