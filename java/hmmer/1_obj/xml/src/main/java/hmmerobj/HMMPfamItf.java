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

package hmmerobj;

import es.bsc.compss.types.annotations.Constraints;
import es.bsc.compss.types.annotations.task.Method;
import es.bsc.compss.types.annotations.Parameter;
import es.bsc.compss.types.annotations.parameter.Direction;
import es.bsc.compss.types.annotations.parameter.Type;


public interface HMMPfamItf {

	@Constraints(storageSize = "0.5")
	@Method(declaringClass = "hmmerobj.HMMPfamImpl")
	String hmmpfam(
		@Parameter(type = Type.STRING, direction = Direction.IN)
		String hmmpfamBin,
		@Parameter(type = Type.STRING, direction = Direction.IN)
		String commandLineArgs,
		@Parameter(type = Type.FILE, direction = Direction.IN)
		String seqFile,
		@Parameter(type = Type.FILE, direction = Direction.IN)
		String dbFile
	);

	@Method(declaringClass = "hmmerobj.HMMPfamImpl")
	String mergeSameDB(
		@Parameter(type = Type.OBJECT, direction = Direction.IN)
		String result1,
		@Parameter(type = Type.OBJECT, direction = Direction.IN)
		String result2
	);

	@Method(declaringClass = "hmmerobj.HMMPfamImpl")
	String mergeSameSeq(
		@Parameter(type = Type.OBJECT, direction = Direction.IN)
		String resultFile1,
		@Parameter(type = Type.OBJECT, direction = Direction.IN)
		String resultFile2,
		@Parameter(type = Type.INT, direction = Direction.IN)
		int aLimit
	);
}
