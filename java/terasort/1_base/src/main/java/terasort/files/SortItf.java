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
package terasort.files;

import es.bsc.compss.types.annotations.Parameter;
import es.bsc.compss.types.annotations.task.Method;
import es.bsc.compss.types.annotations.parameter.Direction;
import es.bsc.compss.types.annotations.parameter.Type;

public interface SortItf {

    @Method(declaringClass = "terasort.files.SortImpl")
    Fragment getFragment(
            @Parameter(type = Type.FILE, direction = Direction.IN) String filePath
    );

    @Method(declaringClass = "terasort.files.SortImpl")
    Fragment filterTask(
            @Parameter Fragment fragment,
            @Parameter long startValue,
            @Parameter long endValue
    );

    @Method(declaringClass = "terasort.files.SortImpl")
    Fragment reduceTask(
            @Parameter Fragment m1,
            @Parameter Fragment m2
    );

    @Method(declaringClass = "terasort.files.SortImpl")
    Fragment sortPartition(
            @Parameter Fragment fragment
    );

    @Method(declaringClass = "terasort.files.SortImpl")
    Integer saveFragment(
            @Parameter Fragment fragment,
            @Parameter String filePath
    );

    @Method(declaringClass = "terasort.files.SortImpl")
    Integer reduceCount(
            @Parameter Integer i1,
            @Parameter Integer i2
    );

}
