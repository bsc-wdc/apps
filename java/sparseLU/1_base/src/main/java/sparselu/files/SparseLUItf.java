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

package sparselu.files;

import es.bsc.compss.types.annotations.task.Method;
import es.bsc.compss.types.annotations.Parameter;
import es.bsc.compss.types.annotations.parameter.Direction;
import es.bsc.compss.types.annotations.parameter.Type;


public interface SparseLUItf {

	@Method(declaringClass = "sparselu.files.SparseLUImpl")
	void lu0(
		@Parameter(type = Type.FILE, direction = Direction.INOUT)
		String diag
	);
	
	@Method(declaringClass = "sparselu.files.SparseLUImpl")
    void bdiv(
    	@Parameter(type = Type.FILE)
    	String diag,
    	@Parameter(type = Type.FILE, direction = Direction.INOUT)
    	String row
    );

	@Method(declaringClass = "sparselu.files.SparseLUImpl")
    void bmod(
    	@Parameter(type = Type.FILE)
    	String row,
    	@Parameter(type = Type.FILE)
    	String col,
    	@Parameter(type = Type.FILE, direction = Direction.INOUT)
    	String inner
    );
	
	@Method(declaringClass = "sparselu.files.SparseLUImpl")
    void fwd(
    	@Parameter(type = Type.FILE)
		String a,
		@Parameter(type = Type.FILE, direction = Direction.INOUT)
		String b
	);

}
