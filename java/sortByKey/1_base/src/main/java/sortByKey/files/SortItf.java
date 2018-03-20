package sortByKey.files;

import es.bsc.compss.types.annotations.Parameter;
import es.bsc.compss.types.annotations.parameter.Direction;
import es.bsc.compss.types.annotations.parameter.Type;
import es.bsc.compss.types.annotations.task.Method;


public interface SortItf {
	
	@Method(declaringClass = "sortByKey.files.SortImpl")
	SortedTreeMap sortPartitionFromFile(
			@Parameter(type = Type.FILE, direction = Direction.IN) String file
	);
	
	@Method(declaringClass = "sortByKey.files.SortImpl")
	SortedTreeMap reduceTask(
			@Parameter SortedTreeMap m1, 
			@Parameter SortedTreeMap m2
	);
	
}
