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
package terasort.filesFilteredShared;

import es.bsc.compss.types.annotations.task.Method;
import es.bsc.compss.types.annotations.Parameter;
import es.bsc.compss.types.annotations.parameter.Direction;
import es.bsc.compss.types.annotations.parameter.Type;

public interface SortItf {

    @Method(declaringClass = "terasort.filesFilteredShared.SortImpl")
    Integer getFilteredFragment(
            @Parameter(type = Type.STRING, direction = Direction.IN) String filePath,
            @Parameter(type = Type.LONG, direction = Direction.IN) long startPos,
            @Parameter(type = Type.LONG, direction = Direction.IN) long endPos,
            @Parameter(type = Type.OBJECT, direction = Direction.IN) String[] bucketsPath,
            @Parameter(type = Type.LONG, direction = Direction.IN) long bucketStep,
            @Parameter(type = Type.INT, direction = Direction.IN) int part
    );

    @Method(declaringClass = "terasort.filesFilteredShared.SortImpl", priority = "true")
    Integer reduceBuckets(
            @Parameter Integer i1,
            @Parameter Integer i2
    );

    @Method(declaringClass = "terasort.filesFilteredShared.SortImpl", priority = "true")
    Long sortBucket(
            @Parameter(type = Type.STRING, direction = Direction.IN) String bucket,
            @Parameter(type = Type.STRING, direction = Direction.IN) String output,
            @Parameter Integer rangeElems
    );

    @Method(declaringClass = "terasort.filesFilteredShared.SortImpl")
    Long reduceSortedBuckets(
            @Parameter Long i1,
            @Parameter Long i2
    );
}
