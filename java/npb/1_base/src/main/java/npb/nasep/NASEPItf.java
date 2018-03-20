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

package npb.nasep;

import es.bsc.compss.types.annotations.task.Method;
import es.bsc.compss.types.annotations.Parameter;
import es.bsc.compss.types.annotations.parameter.Direction;

/**
 *
 * @author flordan
 */
public interface NASEPItf {

        @Method(declaringClass = "npb.nasep.NASEPImpl")
        void reduce (
            @Parameter(direction = Direction.INOUT)
            double[] values,
            @Parameter
            double[] part);

        @Method(declaringClass = "npb.nasep.NASEPImpl")
        double[] generate (
            @Parameter
            int m,
            @Parameter
            int numberProcs,
            @Parameter
            int rank);
        
}
