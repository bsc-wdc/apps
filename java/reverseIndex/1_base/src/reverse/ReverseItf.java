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
package reverse;

import es.bsc.compss.types.annotations.Constraints;
import es.bsc.compss.types.annotations.Method;
import es.bsc.compss.types.annotations.Parameter;
import es.bsc.compss.types.annotations.Parameter.Direction;
import es.bsc.compss.types.annotations.Parameter.Type;

public interface ReverseItf {

	@Method(declaringClass = "reverse.ReverseImpl")
	@Constraints(processorCPUCount = 1, memoryPhysicalSize = 0.5f)
	void parse(@Parameter(type = Type.STRING, direction = Direction.IN) String fileName,
			@Parameter(type = Type.FILE, direction = Direction.IN) String html,
			@Parameter(type = Type.FILE, direction = Direction.OUT) String count);

	@Method(declaringClass = "reverse.ReverseImpl")
	@Constraints(processorCPUCount = 1, memoryPhysicalSize = 0.5f)
	void parseDir(@Parameter(type = Type.STRING, direction = Direction.IN) String directory,
			@Parameter(type = Type.INT, direction = Direction.IN) int begin,
			@Parameter(type = Type.INT, direction = Direction.IN) int end,
			@Parameter(type = Type.FILE, direction = Direction.OUT) String result,
			@Parameter(type = Type.BOOLEAN, direction = Direction.IN) boolean debug);

	@Method(declaringClass = "reverse.ReverseImpl")
	@Constraints(processorCPUCount = 1, memoryPhysicalSize = 0.5f)
	void mergeFiles(@Parameter(type = Type.FILE, direction = Direction.INOUT) String f1,
			@Parameter(type = Type.FILE, direction = Direction.IN) String f2);

	@Method(declaringClass = "reverse.ReverseImpl")
	@Constraints(processorCPUCount = 1, memoryPhysicalSize = 0.5f)
	void mergePackage(@Parameter(type = Type.FILE, direction = Direction.IN) String pack,
			@Parameter(type = Type.FILE, direction = Direction.OUT) String indexes);
}
