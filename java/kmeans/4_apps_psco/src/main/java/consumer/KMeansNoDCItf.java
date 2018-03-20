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
package consumer;

import es.bsc.compss.types.annotations.task.Method;
import es.bsc.compss.types.annotations.Parameter;
import es.bsc.compss.types.annotations.Parameter.Direction;
import es.bsc.compss.types.annotations.Parameter.Type;
import model.Fragment;
import model.SumPoints;


public interface KMeansNoDCItf {

	// @Method(declaringClass = "consumer.KMeansNoDC")
	// public Fragment createFragment(@Parameter(type = Type.INT, direction =
	// Direction.IN) int vectors,
	// @Parameter(type = Type.INT, direction = Direction.IN) int dimensions,
	// @Parameter(type = Type.INT, direction = Direction.IN) int seed);

	@Method(declaringClass = "consumer.KMeansNoDC")
	public SumPoints clusters_points_and_partial_sum(
			@Parameter(type = Type.OBJECT, direction = Direction.IN) Fragment fragment,
			@Parameter(type = Type.OBJECT, direction = Direction.IN) Fragment mu,
			@Parameter(type = Type.INT, direction = Direction.IN) int k,
			@Parameter(type = Type.INT, direction = Direction.IN) int ind);

	// With return
	@Method(declaringClass = "consumer.KMeansNoDC") // , priority = true)
	public SumPoints reduceCentersTask(@Parameter(type = Type.OBJECT, direction = Direction.IN) SumPoints a,
			@Parameter(type = Type.OBJECT, direction = Direction.IN) SumPoints b);

	// @Method(declaringClass = "consumer.KMeansNoDC")
	// public Clusters clusters_points_partial(@Parameter(type = Type.OBJECT,
	// direction = Direction.IN) Fragment points,
	// @Parameter(type = Type.OBJECT, direction = Direction.IN) Fragment mu,
	// @Parameter(type = Type.INT, direction = Direction.IN) int k,
	// @Parameter(type = Type.INT, direction = Direction.IN) int ind);

	// @Method(declaringClass = "consumer.KMeansNoDC")
	// public SumPoints partial_sum(@Parameter(type = Type.OBJECT, direction =
	// Direction.IN) Fragment points,
	// @Parameter(type = Type.OBJECT, direction = Direction.IN) Clusters
	// cluster,
	// @Parameter(type = Type.INT, direction = Direction.IN) int k,
	// @Parameter(type = Type.INT, direction = Direction.IN) int ind);

	// With INOUT
	// @Method(declaringClass = "consumer.KMeansNoDC", priority = true)
	// public void reduceCentersTask(@Parameter(direction = Direction.INOUT)
	// SumPoints a, @Parameter SumPoints b);

}
