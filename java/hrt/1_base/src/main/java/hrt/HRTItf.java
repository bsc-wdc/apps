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

package hrt;

import es.bsc.compss.types.annotations.Constraints;
import es.bsc.compss.types.annotations.task.Method;
import es.bsc.compss.types.annotations.Parameter;
import es.bsc.compss.types.annotations.parameter.Direction;
import es.bsc.compss.types.annotations.parameter.Type;


public interface HRTItf {
	
        @Method(declaringClass = "hrt.HRTImpl")
        @Constraints(computingUnits = "1", memorySize = "1.0")
        void modeling(
        		@Parameter(type = Type.STRING, direction = Direction.IN)
        		String scriptPath,

        		@Parameter(type = Type.FILE, direction = Direction.IN)
        		String confFile,

          		@Parameter(type = Type.STRING, direction = Direction.IN)
                String user,

                @Parameter(type = Type.INT, direction = Direction.IN)
                int index,

                @Parameter(type = Type.FILE, direction = Direction.OUT)
                String modelLog);
        
        @Method(declaringClass = "hrt.HRTImpl")
        @Constraints(computingUnits = "1", memorySize = "1.0")
        void genConfigFile(
        		@Parameter(type = Type.STRING, direction = Direction.IN)
        		String startDate,
        		@Parameter(type = Type.STRING, direction = Direction.IN)
        		String duration,
        		@Parameter(type = Type.FILE, direction = Direction.OUT)
        		String confFile);
        
        @Method(declaringClass = "hrt.HRTImpl")
        @Constraints(computingUnits = "1", memorySize = "1.0")
        void mergeMonitorLogs(
                @Parameter(type = Type.FILE, direction = Direction.INOUT)
                String partialFileA,
                
                @Parameter(type = Type.FILE, direction = Direction.IN)
                String partialFileB);
}
