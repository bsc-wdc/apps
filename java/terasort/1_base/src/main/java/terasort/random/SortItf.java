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
package terasort.random;

import es.bsc.compss.types.annotations.task.Method;
import es.bsc.compss.types.annotations.Parameter;

public interface SortItf {

    @Method(declaringClass = "terasort.random.SortImpl")
    Fragment sortPartition(
            @Parameter Fragment fragment
    );

    @Method(declaringClass = "terasort.random.SortImpl")
    Fragment generateFragment(
            @Parameter int numKeys,
            @Parameter int uniqueKeys,
            @Parameter int keyLength,
            @Parameter int uniqueValues,
            @Parameter int valueLength,
            @Parameter long randomSeed
    );

    @Method(declaringClass = "terasort.random.SortImpl")
    Fragment reduceTask(
            @Parameter Fragment m1,
            @Parameter Fragment m2
    );

    @Method(declaringClass = "terasort.random.SortImpl")
    Fragment filterTask(
            @Parameter Fragment fragment,
            @Parameter int startValue,
            @Parameter int endValue
    );

}
