#!/bin/python

from subprocess import Popen, PIPE
from os import mkdir, listdir
from os.path import isfile, join, exists

from config import *

def read():
    samples = [join(IN_DIR, f) for f in listdir(IN_DIR) if isfile(join(IN_DIR, f))]

    return samples

def checkDirs():
    for dir in IN_DIR, META_DIR, TOOLS_DIR:
        if not exists(dir):
           raise Exception(dir + " does not exist!")

    if not exists(OUT_DIR):
       mkdir(OUT_DIR)

def runCMD(cmd, outputfile=None):
    stdoutHandle = PIPE
    if outputfile != None:
        stdoutHandle = outputfile

    p = Popen(cmd, stdout=stdoutHandle, stderr=PIPE)
    stdout, stderr = p.communicate()

    if "ERROR" in stderr:
        raise Exception(stderr)

    return stdout
