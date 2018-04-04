#!/usr/bin/python

from pycompss.api.parameter import *
from pycompss.api.task import task


def call_cmd(cmd, file_out=None):
    from subprocess import Popen

    print "cmd: ", cmd
    if file_out is None:
        p = Popen(cmd, shell=True)
    else:
        with open(file_out, 'w') as f:
            p = Popen(cmd, shell=True, stdout=f)
    p.wait()


@task(bwa_bin=FILE_IN, ssc_bin=FILE_IN, seq1_file=FILE_IN, seq2_file=FILE_IN, contig_file=FILE_IN, varargsType=FILE_IN,
      returns=dict)
def mapping(bwa_bin, ssc_bin, seq1_file, seq2_file, contig_file, bwa_db_file, *args):
    """Mapping. First part of the workflow.

    :param bwa_bin: Path to the BWA exec file
    :param ssc_bin: Path to the splitSam2Contig exec file
    :param seq1_file: Path to this input file
    :param seq2_file: Path to this input file
    :param contig_file: Path to this input file
    :param bwa_db_file: Path to this input file
    :param args: bwa database files. They must be in the same folder as bwa_db_file
    :return: dictionary
    """

    import errno
    import os
    import os.path

    # print "mapping!!"               # dbg

    num_threads = '1'
    tmp_dir = "ngsa_mini_py_temp" + os.path.splitext(seq1_file)[1]
    # print "abans de fer el dir"     # dbg
    try:
        os.makedirs(tmp_dir, mode=0777)
    except OSError as e:
        if e.errno != errno.EEXIST:
            print "Failed to create Directory[{}].\n".format(tmp_dir)
            raise
        else:
            print "File already exists"
    # print "despres de fer el dir"   # dbg

    cmd = bwa_bin + ' aln -t ' + num_threads + ' ' + bwa_db_file + ' ' + seq1_file
    seq1_sai = tmp_dir + '/' + os.path.basename(seq1_file) + ".sai"
    call_cmd(cmd, seq1_sai)
    # print "executat 1"  # dbg

    cmd = bwa_bin + ' aln -t ' + num_threads + ' ' + bwa_db_file + ' ' + seq2_file
    seq2_sai = tmp_dir + '/' + os.path.basename(seq2_file) + ".sai"
    call_cmd(cmd, seq2_sai)
    # print "executat 2"  # dbg

    cmd = bwa_bin + ' sampe ' + bwa_db_file + ' ' + seq1_sai + ' ' + seq2_sai + ' ' + seq1_file + ' ' + seq2_file
    sam_file = tmp_dir + '/' + "0.sam"
    call_cmd(cmd, sam_file)
    # print "executat 3"  # dbg

    cmd = ssc_bin + ' ' + contig_file + ' ' + sam_file + ' ' + tmp_dir
    call_cmd(cmd)       # in the original version, stdout is redirected to stderr
    # print "executat 4"  # dbg

    # print "creating the dictionary"                                 # dbg
    dic = {}
    for contig in os.listdir(tmp_dir):
        # print "contig file: " + contig                              # dbg
        if os.path.splitext(contig)[1] == '.sam' and contig != 'single.sam' and \
                            contig != 'unmap.sam' and contig != '0.sam':
            # print "  ------> writing to dictionary"                 # dbg
            with open(tmp_dir + '/' + contig, 'r') as f:
                dic[contig] = f.read()
    # print "dictionary created with length: " + str(len(dic))        # dbg
    return dic


@task(d1=INOUT)
def merge(d1, d2):
    # type: (dict, dict) -> None
    """ Joins dictionaries d1 and d2 into d1. If a key k appears in both dictionaries with their values being v1 and v2
     respectively, the joined dictionary d has d[k]==v1+v2. d1 is extended

    :param d1: dictionary. It is extended
    :param d2: dictionary
    """

    # print "merging dictionary d2 into d1. List of keys from d2:"    # dbg

    for k in d2:
        if d1.has_key(k):
            # print "    " + k + ": updated in d1"                    # dbg
            d1[k] = d1[k] + d2[k]
        else:
            # print "    " + k + ": added into d1"                    # dbg
            d1[k] = d2[k]


def mapping_merge(d1, cmd_dir, bwa_db_file, contig_file, seq_files):
    # type: (dict, str, str, str, list) -> dict
    """ Performs the mapping and merge operations of the workflow.

    :param d1: Must be an empty dictionary which will be overwritten with the returned dictionary
    :param cmd_dir: path to the command directory
    :param bwa_db_file: path to BWA DB file
    :param contig_file: path to contig file
    :param seq_files: path to input sequence files
    :return: returns the d1 dictionary, which has had the contigs from seq_files merged into it
    """

    import os.path
    bwa_db_dir = os.path.dirname(bwa_db_file)
    bwa_db_files = [bwa_db_dir + '/' + x for x in os.listdir(os.path.split(bwa_db_file)[0])]

    # print "mapping and merge from files: ", str(seq_files)  # dbg

    bwa_bin = cmd_dir + "/bwa"
    ssc_bin = cmd_dir + "/splitSam2Contig2"

    # print "Calling mapping with parameters:"                # dbg
    # print "bwa_bin=", bwa_bin                               # dbg
    # print "ssc_bin=", ssc_bin                               # dbg
    # print "seq_files[0]=", seq_files[0]                     # dbg
    # print "seq_files[1]=", seq_files[1]                     # dbg
    # print "contig_file=", contig_file                       # dbg
    # print "bwa_db_file=", bwa_db_file                       # dbg
    # print "bwa_db_files=", str(bwa_db_files)                # dbg
    d2 = mapping(bwa_bin, ssc_bin, seq_files[0], seq_files[1], contig_file, bwa_db_file, *bwa_db_files)

    merge(d1, d2)

    return d1

# NUM_BUCKETS = 50
# @task(returns=list)
# def split(contigs):
#     buckets = [{} for _ in range(NUM_BUCKETS)]
#     i = 0
#     for c in contigs:
#         buckets[i][c] = contigs[c]
#         i = (i + 1) % NUM_BUCKETS
#     return buckets


@task(samtools_bin=FILE_IN, snp_bin=FILE_IN, ref_idx_file=FILE_IN, ref_file=FILE_IN, out_file1=FILE_OUT,
      out_file2=FILE_OUT, out_file3=FILE_OUT)
def rmdup_analyze(samtools_bin, snp_bin, ref_idx_file, ref_file, contig_sam, contig_content, out_file1, out_file2,
                  out_file3):
    # type: (str, str, str, str, str, str, str, str) -> None
    """ Performs the remove duplicates and analyze tasks to the contig_sam

    :param samtools_bin: path to the samtools exec file
    :param snp_bin: path to the SNP exec file
    :param ref_idx_file: path to reference index file
    :param ref_file: path to reference file
    :param contig_sam: '<contig>.sam'
    :param contig_content: content of the result of the merge of the mapping operations to this contig
    :param out_file1: path to '<contig>.indel' file, where output will be written
    :param out_file2: path to '<contig>.snp' file, where output will be written
    :param out_file3: path to '<contig>.sum' file, where output will be written
    """

    import errno
    import os

    # rm_dup
    n_memory = '800000000'
    tmp_dir = "ngsa_mini_py_temp"
    try:
        os.makedirs(tmp_dir, mode=0700)
    except OSError as e:
        if e.errno != errno.EEXIST:
            print "Failed to create Directory[{}].\n".format(tmp_dir)
            raise

    contig = os.path.splitext(contig_sam)[0]

    # Write sam into file
    sam_file = tmp_dir + '/' + contig + '.sam'
    with open(sam_file, 'w') as f:
        f.write(contig_content)

    bam_file = tmp_dir + '/' + contig + '.bam'
    cmd = samtools_bin + ' import ' + ref_idx_file + ' ' + sam_file + ' ' + bam_file
    call_cmd(cmd)

    sort_file = tmp_dir + '/' + contig + '.sort'
    cmd = samtools_bin + ' sort -m ' + n_memory + ' ' + bam_file + ' ' + sort_file
    call_cmd(cmd)

    rmdup_file = tmp_dir + '/' + contig + '.sort.rmdup.bam'
    cmd = samtools_bin + ' rmdup ' + sort_file + '.bam ' + rmdup_file
    call_cmd(cmd)

    # analyze
    pileup_file = tmp_dir + '/' + contig + '.pile'
    cmd = samtools_bin, ' pileup -s -cf' + ' ' + ref_file + ' ' + rmdup_file
    call_cmd(cmd, pileup_file)

    cmd = snp_bin + ' -INF ' + pileup_file + ' -INDEL ' + out_file1 + ' -SNP ' + out_file2 + ' -SUM ' + out_file3
    call_cmd(cmd)       # in the original version, stdout is redirected to stderr


@task(tar_file=FILE_INOUT, file1=FILE_IN, file2=FILE_IN, file3=FILE_IN)
def tar(tar_file, file1, file2, file3):
    # type: (str, str, str, str) -> None
    """ Append three files into a tar file

    :param tar_file: path to tar file (it will be overwritten with the output file)
    :param file1: path to the first file to be appended
    :param file2: path to the second file to be appended
    :param file3: path to the third file to be appended
    :return: path to tar file (same as out_tar)
    """
    import tarfile
    '''
    f = tarfile.open(tar_file, 'a')
    f.add(file1)
    f.add(file2)
    f.add(file3)
    f.close()
    '''
    t = tarfile.TarFile(tar_file, dereference=True, mode='a')
    t.add(file1)
    t.add(file2)
    t.add(file3)
    t.close()


def rmdup_analyze_tar(cmd_dir, ref_idx_file, ref_file, contig_sam, contig_content, tar_file):
    # type: (str, str, str, str, str, str) -> str
    """ Remove duplicates and analyze the contig, then append the result to the tar file

    :param cmd_dir: path to the command directory
    :param ref_idx_file: path to reference index file
    :param ref_file: path to reference file
    :param contig_sam: '<contig>.sam'
    :param contig_content: content of the result of the merge of the mapping operations to this contig
    :param tar_file: path to a .tar file to which the results of the analysis to this contig will be applied
    :return: (future object of a string) path to the output .tar file (same as tar_file)
    """

    import os.path
    from pycompss.api.api import compss_delete_file
    '''
    import os
    import errno

    tmp_dir = "ngsa_mini_py_temp"
    try:
        os.makedirs(tmp_dir, mode=0700)
    except OSError as e:
        if e.errno != errno.EEXIST:
            print "Failed to create Directory[{}].\n".format(tmp_dir)
            raise
    '''

    # Remove duplicates (rmdup) and analyze
    samtools_bin = cmd_dir + "/samtools"
    snp_bin = cmd_dir + "/snp"
    contig = os.path.splitext(contig_sam)[0]
    out_file1 = "ngsa_mini_py_temp/" + contig + '.indel'
    out_file2 = "ngsa_mini_py_temp/" + contig + '.snp'
    out_file3 = "ngsa_mini_py_temp/" + contig + '.sum'
    # print "Calling rmdup_analyze with parameters:"               # dbg
    # print "samtools_bin=", samtools_bin                          # dbg
    # print "snp_bin=", snp_bin                                    # dbg
    # print "ref_idx_file=", ref_idx_file                          # dbg
    # print "ref_file=", ref_file                                  # dbg
    # print "contig_sam=", contig_sam                              # dbg
    # print "str(len(contig_content))=", str(len(contig_content))  # dbg
    # print "out_file1=", out_file1                                # dbg
    # print "out_file2=", out_file2                                # dbg
    # print "out_file3=", out_file3                                # dbg
    rmdup_analyze(samtools_bin, snp_bin, ref_idx_file, ref_file, contig_sam, contig_content, out_file1, out_file2,
                  out_file3)

    # tar files
    # print "Calling tar with parameters:"                         # dbg
    # print "tar_file=", tar_file                                  # dbg
    # print "out_file1=", out_file1                                # dbg
    # print "out_file2=", out_file2                                # dbg
    # print "out_file3=", out_file3                                # dbg
    tar(tar_file, out_file1, out_file2, out_file3)
    compss_delete_file(out_file1)
    compss_delete_file(out_file2)
    compss_delete_file(out_file3)
    return tar_file


def init_tar(out_dir):
    import tarfile
    tar_file = out_dir + '/output.tar'
    f = tarfile.open(tar_file, 'a')
    f.close()
    return tar_file


def mergeReduce(function, data, init):
    """ Apply function cumulatively to the items of data,
        from left to right in binary tree structure, so as to
        reduce the data to a single value.
    :param function: function to apply to reduce data
    :param data: List of items to be reduced
    :return: result of reduce the data to a single value
    """
    from collections import deque
    q = deque(xrange(len(data)))
    while len(q):
        x = q.popleft()
        if len(q):
            y = q.popleft()
            data[x] = function(data[x], data[y])
            q.append(x)
        else:
            return data[x]


def main():
    import errno
    import os
    import sys
    import time
    from pycompss.api.api import barrier, compss_wait_on

    # usage
    if len(sys.argv) != 8:
        print "Usage: {} BWA_DB_FILE CONTIG_FILE REFERENCE_FILE REFERENCE_INDEX_FILE INPUT_DIR WORK_DIR " \
              "NUM_PROCESSES\n\n" \
              "Program name must be called with an absolute path (starting with '/').".format(sys.argv[0])
        return 1

    # find program directory and basenames
    cmd_dir = os.path.dirname(sys.argv[0])
    if cmd_dir == "" or cmd_dir[0] != '/':
        print "Program must be called with an absolute path (starting with '/')"
        return 1
    prog_basename = os.path.basename(os.path.splitext(sys.argv[0])[0])

    # read inputs
    bwa_db_file = sys.argv[1]
    contig_file = sys.argv[2]
    ref_file = sys.argv[3]
    ref_idx_file = sys.argv[4]
    in_dir_prefix = sys.argv[5]
    work_dir = sys.argv[6]
    num_processes = int(sys.argv[7])

    # setup directories
    in_dirs = [in_dir_prefix + '/' + str(x) for x in range(num_processes)]

    out_dir = "{}/{}_OUT".format(work_dir, prog_basename)
    try:
        os.makedirs(out_dir, mode=0700)
    except OSError as e:
        if e.errno != errno.EEXIST:
            print "Failed to create Directory[{}].\n".format(out_dir)
            raise

    start_time = time.time()

    # mapping & merge
    inputs = []
    for in_dir in in_dirs:
        exts = set(map(lambda file1: os.path.splitext(file1)[1], os.listdir(in_dir)))
        for ext in exts:
            elem = [in_dir + '/' + f for f in os.listdir(in_dir) if f.endswith(ext)]
            elem.sort()
            inputs.append(elem)
    # inputs = [[in_dir+'part_1.'+i, in_dir+'part_2.'+i] for i in range(num_processes)]
    #        ~ [[part_1.0, part_2.0], [part_1.1, part_2.1], ...]

    # print "Inputs: ", str(inputs)               # dbg
    contigs = reduce(lambda e1, e2: mapping_merge(e1, cmd_dir, bwa_db_file, contig_file, e2), inputs, {})
    # print "before compss_wait_on"               # dbg
    contigs = compss_wait_on(contigs)
    # print "after compss_wait_on"                # dbg
    # with open('output.dict', 'w') as f:         # dbg
    #     f.write(str(contigs))                   # dbg
    # buckets = split(contigs)

    # rm_dup & analyze
    tar_file = init_tar(out_dir)
    reduce(lambda tar_file1, contig_sam: rmdup_analyze_tar(cmd_dir, ref_idx_file, ref_file, contig_sam,
                                                           contigs[contig_sam], tar_file1),
           contigs, tar_file)
    # print "before barrier"                      # dbg
    barrier()
    # print "after barrier"                       # dbg

    print "NGSA-mini-py with {} processes. Ellapsed Time {} (s)".format(num_processes, time.time() - start_time)


if __name__ == "__main__":
    main()