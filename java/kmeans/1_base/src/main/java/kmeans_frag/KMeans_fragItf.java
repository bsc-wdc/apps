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
package kmeans_frag;

import es.bsc.compss.types.annotations.Parameter;
import es.bsc.compss.types.annotations.task.Method;


public interface KMeans_fragItf {
	
    @Method(declaringClass = "kmeans_frag.KMeans_frag")
    Fragment generateFragment(
    		@Parameter int numV,
    		@Parameter int nDimensions,
                @Parameter int seed,
                @Parameter int scale
    );
    
    @Method(declaringClass = "kmeans_frag.KMeans_frag")
    Clusters clusters_points_partial(
    		@Parameter Fragment points,
		@Parameter Fragment mu,
		@Parameter int k,
		@Parameter int ind
    );

    @Method(declaringClass = "kmeans_frag.KMeans_frag")
    SumPoints partial_sum(
		@Parameter Fragment points,
		@Parameter Clusters cluster,
		@Parameter int k,
		@Parameter int ind
    );
 
    @Method(declaringClass = "kmeans_frag.KMeans_frag", priority = "true")
    SumPoints reduceCentersTask(
    		@Parameter SumPoints a,
		@Parameter SumPoints b
    );
}
