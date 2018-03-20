package sortByKey.random;

import java.util.HashMap;

import es.bsc.compss.types.annotations.Parameter;
import es.bsc.compss.types.annotations.task.Method;



public interface SortItf {
	
	@Method(declaringClass = "sortByKey.random.SortImpl")
	SortedTreeMap sortPartition(
			@Parameter HashMap<String,String> fragment
	);
		
	@Method(declaringClass = "sortByKey.random.SortImpl")
	HashMap<String, String> generateFragment(
			@Parameter int numKeys, 
			@Parameter int uniqueKeys, 
			@Parameter int keyLength, 
			@Parameter int uniqueValues,
			@Parameter int valueLength, 
			@Parameter long randomSeed
	);
	
	@Method(declaringClass = "sortByKey.random.SortImpl")
	SortedTreeMap reduceTask(
			@Parameter SortedTreeMap m1, 
			@Parameter SortedTreeMap m2
	);
		
}
