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
package discrete;

import es.bsc.compss.types.annotations.Parameter;
import es.bsc.compss.types.annotations.parameter.Direction;
import es.bsc.compss.types.annotations.parameter.Type;
import es.bsc.compss.types.annotations.task.Method;

public interface DiscreteItf {

	@Method(declaringClass = "discrete.DiscreteImpl")
	void genReceptorLigand(@Parameter(type = Type.FILE, direction = Direction.IN) String pdbFile,
			@Parameter(type = Type.STRING, direction = Direction.IN) String binDir,
			@Parameter(type = Type.FILE, direction = Direction.OUT) String recFile,
			@Parameter(type = Type.FILE, direction = Direction.OUT) String ligFile);

	@Method(declaringClass = "discrete.DiscreteImpl")
	void dmdSetup(@Parameter(type = Type.FILE, direction = Direction.IN) String recFile,
			@Parameter(type = Type.FILE, direction = Direction.IN) String ligFile,
			@Parameter(type = Type.STRING, direction = Direction.IN) String binDir,
			@Parameter(type = Type.STRING, direction = Direction.IN) String dataDir,
			@Parameter(type = Type.FILE, direction = Direction.OUT) String topFile,
			@Parameter(type = Type.FILE, direction = Direction.OUT) String crdFile);

	@Method(declaringClass = "discrete.DiscreteImpl")
	void simulate(@Parameter(type = Type.FILE, direction = Direction.IN) String paramFile,
			@Parameter(type = Type.FILE, direction = Direction.IN) String topFile,
			@Parameter(type = Type.FILE, direction = Direction.IN) String crdFile,
			@Parameter(type = Type.STRING, direction = Direction.IN) String natom,
			@Parameter(type = Type.STRING, direction = Direction.IN) String binDir,
			@Parameter(type = Type.STRING, direction = Direction.IN) String dataDir,
			@Parameter(type = Type.FILE, direction = Direction.OUT) String averageFile);

	@Method(declaringClass = "discrete.DiscreteImpl")
	void merge(@Parameter(type = Type.FILE, direction = Direction.INOUT) String f1,
			@Parameter(type = Type.FILE, direction = Direction.IN) String f2);

	@Method(declaringClass = "discrete.DiscreteImpl")
	void evaluate(@Parameter(type = Type.FILE, direction = Direction.IN) String averageFile,
			@Parameter(type = Type.FILE, direction = Direction.IN) String pydockFile,
			@Parameter(type = Type.DOUBLE, direction = Direction.IN) double fvdw,
			@Parameter(type = Type.DOUBLE, direction = Direction.IN) double fsolv,
			@Parameter(type = Type.DOUBLE, direction = Direction.IN) double eps,
			@Parameter(type = Type.FILE, direction = Direction.OUT) String scoreFile,
			@Parameter(type = Type.FILE, direction = Direction.OUT) String coeffFile

	);

	@Method(declaringClass = "discrete.DiscreteImpl")
	void min(@Parameter(type = Type.FILE, direction = Direction.INOUT) String f1,
			@Parameter(type = Type.FILE, direction = Direction.IN) String f2);
}
