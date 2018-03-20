/*
 *  Copyright 2002-2016 Barcelona Supercomputing Center (www.bsc.es)
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
package terasort.filesFilteredShared2;

import es.bsc.compss.types.annotations.task.Method;
import es.bsc.compss.types.annotations.Parameter;
import es.bsc.compss.types.annotations.parameter.Direction;
import es.bsc.compss.types.annotations.parameter.Type;

public interface SortItf {

    //@Constraints(memoryPhysicalSize = 16.0f)
    @Method(declaringClass = "terasort.filesFilteredShared2.SortImpl")
    Fragment readBlock(
            @Parameter(type = Type.STRING, direction = Direction.IN) String filePath,
            @Parameter(type = Type.LONG, direction = Direction.IN) long startPos,
            @Parameter(type = Type.LONG, direction = Direction.IN) long endPos
    );

    @Method(declaringClass = "terasort.filesFilteredShared2.SortImpl") //, priority=true)
    Fragment extractSortedBucket(
            @Parameter Fragment f,
            @Parameter(type = Type.LONG, direction = Direction.IN) long startBucket,
            @Parameter(type = Type.LONG, direction = Direction.IN) long endBucket
    );

    @Method(declaringClass = "terasort.filesFilteredShared2.SortImpl")
    Fragment mergeBuckets(
            @Parameter Fragment f1,
            @Parameter Fragment f2
    );

    //@Constraints(memoryPhysicalSize = 16.0f)
    @Method(declaringClass = "terasort.filesFilteredShared2.SortImpl", priority = "true")
    Long saveFragment(
            @Parameter Fragment f,
            @Parameter(type = Type.STRING, direction = Direction.IN) String filePath
    );

    @Method(declaringClass = "terasort.filesFilteredShared2.SortImpl")
    Long reduceSortedBucketsCount(
            @Parameter Long i1,
            @Parameter Long i2
    );

}
