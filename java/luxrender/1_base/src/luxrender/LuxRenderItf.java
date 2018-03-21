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

package luxrender;

import es.bsc.compss.types.annotations.Constraints;
import es.bsc.compss.types.annotations.Method;
import es.bsc.compss.types.annotations.Parameter;
import es.bsc.compss.types.annotations.Parameter.Direction;
import es.bsc.compss.types.annotations.Parameter.Type;

public interface LuxRenderItf {
	
    @Method(declaringClass = "luxrender.LuxRenderImpl")
    @Constraints(processorCPUCount = 4, memoryPhysicalSize = 1.5f)
    void renderPartition(
    		@Parameter(type = Type.STRING, direction = Direction.IN)
    		String inputModelFullPathName,

            @Parameter(type = Type.FILE, direction = Direction.OUT)
            String partialOutputName,

            @Parameter(type = Type.STRING, direction = Direction.IN)
            String luxRenderBinary);
    
    @Method(declaringClass = "luxrender.LuxRenderImpl")
    @Constraints(processorCPUCount = 4, memoryPhysicalSize = 1.5f)
    void mergePartitions(
    		@Parameter(type = Type.STRING, direction = Direction.IN)
    		String luxRenderBinary,

            @Parameter(type = Type.FILE, direction = Direction.IN)
            String a,
            
            @Parameter(type = Type.FILE, direction = Direction.IN)
            String b,

            @Parameter(type = Type.FILE, direction = Direction.OUT)
            String mergedFile);
}
