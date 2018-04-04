#
#  Copyright 2002-2015 Barcelona Supercomputing Center (www.bsc.es)
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

from  active_worker.task import task
from task_types import TaskTypes as tt

_task_full_name = "tejedor_neuron_correlation_pycompss"
_task_caption = "Neuron cross-correlation task - PyCOMPSs version"
_task_author = "tejedor"
_task_description = "Program that computes the cross-correlation between pairs of neurons using PyCOMPSs"
_task_categories = ['test']
_task_compatible_queues = ['all']

from nsdataproc.ns_data_proc_objects import ns_data_processing 

ns_mime_type = 'application/unknown'

from pycompss.runtime.launch import pycompss_launch

@task(accepts=(tt.URIType('application/unknown'),tt.LongType), returns=(tt.URIType('application/unknown'),tt.URIType('application/unknown')))
def tejedor_neuron_correlation(fspikes_uri, num_frags):
    # Stage spike data
    fspikes = tejedor_neuron_correlation.task.uri.get_file(fspikes_uri)

    # Launch the computation
    #(cc_orig, cc_surrs) = ns_data_processing(fspikes, num_frags)
    result = pycompss_launch(app = ns_data_processing, args = (fspikes, num_frags), kwargs = {})
    cc_orig = result[0]
    cc_surrs = result[1]

    # Save originals and surrogates
    cc_orig_uri  = tejedor_neuron_correlation.task.uri.save_file(mime_type=ns_mime_type, src_path=cc_orig, dst_path='result_cc_originals.dat')
    cc_surrs_uri = tejedor_neuron_correlation.task.uri.save_file(mime_type=ns_mime_type, src_path=cc_surrs, dst_path='result_cc_surrogates.dat')
    
    return (cc_orig_uri, cc_surrs_uri)

if __name__ == '__main__':
    import sys
    tejedor_neuron_correlation(tt.URI(ns_mime_type, sys.argv[1]), int(sys.argv[2]))
