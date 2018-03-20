/*
 *  Copyright 2002-2015 Barcelona Supercomputing Center (www.bsc.es)
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
package npb.nasis;

import es.bsc.compss.types.annotations.task.Method;
import es.bsc.compss.types.annotations.Parameter;
import es.bsc.compss.types.annotations.parameter.Direction;


public interface NASISItf {   
    @Method(declaringClass = "npb.nasis.NASISImpl")
    int[] create_seq(
        @Parameter
        double seed,
        @Parameter
        double a,
        @Parameter
        int rank,
        @Parameter
        ISProblemClass ISProblem
    );

    @Method(declaringClass = "npb.nasis.NASISImpl")
    int[] rank0(
        @Parameter
        ISProblemClass ISProblem,
        @Parameter
        int iteration,
        @Parameter(direction = Direction.INOUT)
        int[] key_array,
        @Parameter(direction = Direction.INOUT)
        int[] key_buff1
    );

    @Method(declaringClass = "npb.nasis.NASISImpl")
    int[] rank(
        @Parameter
        ISProblemClass ISProblem,
        @Parameter
        int iteration,
        @Parameter
        int rank,
        @Parameter
        int[] key_array,
        @Parameter(direction = Direction.INOUT)
        int[] key_buff1
    );

    @Method(declaringClass = "npb.nasis.NASISImpl")
    int[] reduceBucketSize(
        @Parameter
        ISProblemClass ISProblem,
        @Parameter
        int[] bucket_size1,
        @Parameter
        int[] bucket_size2
    );
    
    @Method(declaringClass = "npb.nasis.NASISImpl")
    void prepareSend(
        @Parameter
        ISProblemClass ISProblem,
        @Parameter(direction = Direction.INOUT)
        int[] process_bucket_distrib_ptr1,
        @Parameter(direction = Direction.INOUT)
        int[] process_bucket_distrib_ptr2,
        @Parameter
        int[] bucket_size_totals,
        @Parameter
        int[] bucket_size,
        @Parameter(direction = Direction.INOUT)
        int[] send_count
    );

    @Method(declaringClass = "npb.nasis.NASISImpl")
    void transferKeys(
        @Parameter
        int[] key_buff1,
        @Parameter(direction = Direction.INOUT)
        int[] key_buff20,
        @Parameter(direction = Direction.INOUT)
        int[] key_buff21,
        @Parameter(direction = Direction.INOUT)
        int[] key_buff22,
        @Parameter(direction = Direction.INOUT)
        int[] key_buff23,
        @Parameter(direction = Direction.INOUT)
        int[] recv_displ0,
        @Parameter(direction = Direction.INOUT)
        int[] recv_displ1,
        @Parameter(direction = Direction.INOUT)
        int[] recv_displ2,
        @Parameter(direction = Direction.INOUT)
        int[] recv_displ3);
    

    @Method(declaringClass = "npb.nasis.NASISImpl")
    void rank_end(
            @Parameter
            ISProblemClass ISProblem,
            @Parameter
            int rank,
            @Parameter
            int[] process_bucket_distrib_ptr1,
            @Parameter
            int[] process_bucket_distrib_ptr2,
            @Parameter(direction = Direction.INOUT)
            int[] key_buff1,
            @Parameter
            int[] key_buff2,
            @Parameter
            int[] bucket_size_totals,
            @Parameter
            int iteration,
            @Parameter(direction = Direction.INOUT)
            int[] verifies,
            @Parameter(direction = Direction.INOUT)
            int[] total_local_keys
    );


    @Method(declaringClass = "npb.nasis.NASISImpl")
    int[] get_keypart(
        @Parameter
        int[] key_buff10,
        @Parameter
        int[] key_buff11,
        @Parameter
        int[] key_buff12,
        @Parameter
        int[] key_buff13,
        @Parameter
        int[] send_count0,
        @Parameter
        int[] send_count1,
        @Parameter
        int[] send_count2,
        @Parameter
        int[] send_count3,
        @Parameter
        int receiver
    );


    @Method(declaringClass = "npb.nasis.NASISImpl")
    int synchronize(
        @Parameter
        int[] value
    );

    @Method(declaringClass = "npb.nasis.NASISImpl")
    int[] allocIntArray(
        @Parameter
        int buffersSize
    );
    
    @Method(declaringClass = "npb.nasis.NASISImpl")
    void sortKeys(
        @Parameter
        int[] total_local_keys,
        @Parameter(direction = Direction.INOUT)
        int[] key_buff1,
        @Parameter
        int[] key_buff2,
        @Parameter(direction = Direction.INOUT)
        int[] key_array
    );
    
}
