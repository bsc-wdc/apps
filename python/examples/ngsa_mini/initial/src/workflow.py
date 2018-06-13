#!/usr/bin/python

from pycompss.api.parameter import *
from pycompss.api.task import task


def setup_directories(prog_basename, in_dir_prefix, num_processes, global_med_dir, work_dir):
    """

    :rtype: (list, list, str)
    """
    import os
    import errno
    from pycompss.api.api import compss_barrier

    # create med_dirs and check that in_dirs exist
    # print "dbg: creating med_dirs and checking that in_dirs exist..."
    med_dir_prefix = "{}/{}_MED".format(work_dir, prog_basename)
    try:
        os.makedirs(med_dir_prefix, mode=0700)
    except OSError as e:
        if e.errno != errno.EEXIST:
            print "Failed to create Directory[{}].\n".format(med_dir_prefix)
            raise

    in_dirs = ['' for _ in range(num_processes)]
    med_dirs = ['' for _ in range(num_processes)]
    for rank in range(num_processes):
        # check in_dir exists
        in_dir = in_dir_prefix + '/' + str(rank)
        if not os.path.isdir(in_dir):
            print "Input files should be located under rank-named directory, like \"{}\"\n".format(in_dir)
        in_dirs[rank] = in_dir

        # create med_dir
        med_dir = med_dir_prefix + '/' + str(rank)
        try:
            os.makedirs(med_dir, mode=0700)
        except OSError as e:
            if e.errno != errno.EEXIST:  # for some reason this 'if' is not checked in original code
                print "Failed to create Directory[{}].\n".format(med_dir)
                raise
        med_dirs[rank] = med_dir

    # create out_dir
    # print "dbg: creating out_dir..."
    out_dir = "{}/{}_OUT".format(work_dir, prog_basename)
    try:
        os.makedirs(out_dir, mode=0700)
    except OSError as e:
        if e.errno != errno.EEXIST:
            print "Failed to create Directory[{}].\n".format(out_dir)
            raise

    # create global_med_dir
    # print "dbg: creating global_med_dir..."
    try:
        os.makedirs(global_med_dir, mode=0700)
    except OSError as e:
        if e.errno != errno.EEXIST:
            print "Failed to create Directory[{}].\n".format(global_med_dir)
            raise

    return in_dirs, med_dirs, out_dir


# TODO: cleanup all directories. MED & OUT too. Also check that future objects don't cause bugs
def cleanup_directories(global_med_dir):
    import shutil
    try:
        shutil.rmtree(global_med_dir)
    except OSError:
        print "Failed to delete Directory[{}].\n".format(global_med_dir)
        raise


@task(returns=str)
def mapping(cmd_dir, in_dir, med_dir, bwa_db_file, contig_file):
    import subprocess
    cmd = ["{}/workflow_01.sh".format(cmd_dir), in_dir, med_dir, bwa_db_file, contig_file]
    try:
        print str(subprocess.check_output(cmd))
    except subprocess.CalledProcessError, e:
        print "workflow01 stdout output:\n", e.output
    return med_dir


# Returns an int so that it is not ignored by compss_barrier
@task(returns=int)
def merge_step_1(cmd_dir, rank, med_dir, global_med_dir):
    import subprocess
    cmd = ["{}/workflow_02.sh".format(cmd_dir), rank, med_dir, global_med_dir]
    try:
        print str(subprocess.check_output(cmd))
    except subprocess.CalledProcessError, e:
        print "workflow02 stdout output:\n", e.output
    return 0


@task(returns=str)
def merge_step_2(cmd_dir, rank, num_processes, med_dir, global_med_dir):
    import subprocess
    cmd = ["{}/workflow_03.sh".format(cmd_dir), rank, num_processes, med_dir, global_med_dir]
    try:
        print str(subprocess.check_output(cmd))
    except subprocess.CalledProcessError, e:
        print "workflow03 stdout output:\n", e.output
    return med_dir


@task(returns=str)
def rm_dup(cmd_dir, med_dir, ref_idx_file):
    import subprocess
    cmd = ["{}/workflow_04.sh".format(cmd_dir), med_dir, ref_idx_file]
    try:
        print str(subprocess.check_output(cmd))
    except subprocess.CalledProcessError, e:
        print "workflow04 stdout output:\n", e.output
    return med_dir


@task(returns=str)
def analyze(cmd_dir, med_dir, out_dir, ref_file):
    import subprocess
    cmd = ["{}/workflow_05.sh".format(cmd_dir), med_dir, out_dir, ref_file]
    try:
        print str(subprocess.check_output(cmd))
    except subprocess.CalledProcessError, e:
        print "workflow05 stdout output:\n", e.output
    return out_dir


def main():
    import os
    import sys
    import time
    from pycompss.api.api import compss_barrier

    if len(sys.argv) != 8:
        print "Usage: {} BWA_DB_FILE CONTIG_FILE REFERENCE_FILE REFERENCE_INDEX_FILE INPUT_DIR WORK_DIR " \
              "NUM_PROCESSES\n\n" \
              "Program name must be called with an absolute path (starting with '/').".format(sys.argv[0])
        return 1

    # find program directory and basenames
    # print "dbg: finding program directory and basenames..."
    cmd_dir = os.path.dirname(sys.argv[0])
    if cmd_dir == "" or cmd_dir[0] != '/':
        print "Program must be called with an absolute path (starting with '/')"
        return 1
    prog_basename = os.path.basename(os.path.splitext(sys.argv[0])[0])
    # print "dbg: reading inputs..."

    # read inputs
    bwa_db_file = sys.argv[1]
    contig_file = sys.argv[2]
    ref_file = sys.argv[3]
    ref_idx_file = sys.argv[4]
    in_dir_prefix = sys.argv[5]
    work_dir = sys.argv[6]
    num_processes = int(sys.argv[7])

    # print "dbg: setting up directories..."

    # setup directories
    global_med_dir = work_dir + "/SH_GLOBAL"
    (in_dirs, med_dirs, out_dir) = \
        setup_directories(prog_basename, in_dir_prefix, num_processes, global_med_dir, work_dir)
    # MPI_BARRIER

    start_time = time.time()

    # print "dbg: calling workflow scripts..."
    # mapping
    med_dirs = map(lambda rank: mapping(cmd_dir, in_dirs[rank], med_dirs[rank], bwa_db_file, contig_file),
                   range(num_processes))

    # merge
    map(lambda rank: merge_step_1(cmd_dir, str(rank), med_dirs[rank], global_med_dir), range(num_processes))
    compss_barrier()
    # MPI_BARRIER
    med_dirs = map(lambda rank: merge_step_2(cmd_dir, str(rank), str(num_processes), med_dirs[rank], global_med_dir),
                   range(num_processes))

    # rm_dup
    med_dirs = map(lambda rank: rm_dup(cmd_dir, med_dirs[rank], ref_idx_file), range(num_processes))

    # analyze
    map(lambda rank: analyze(cmd_dir, med_dirs[rank], out_dir, ref_file), range(num_processes))
    # MPI_BARRIER
    compss_barrier()

    print "NGSA-mini-py with {} processes. Ellapsed Time {} (s)".format(num_processes, time.time() - start_time)

    cleanup_directories(global_med_dir)


if __name__ == "__main__":
    main()
