package sortByKey.files;

import java.io.File;
import java.util.Arrays;
import java.util.LinkedList;


public class Sort {
	private static int NUM_FRAGMENTS;
	private static String DATA_FOLDER;
	
	private static String[] filePaths;
	private static SortedTreeMap[] partialSorted;
	private static SortedTreeMap result;
	
	public static void main(String args[]) {
		// Get parameters
		if (args.length != 1) {
			System.out.println("[ERROR] Usage: Sort <DATA_FOLDER>");
			System.exit(-1);
		}
		DATA_FOLDER = args[0];
		System.out.println("DATA_FOLDER parameter value = " + DATA_FOLDER);
		
		// Run sort by key app
		long startTime = System.currentTimeMillis();
		Sort sbk = new Sort();
		sbk.run();
		
		System.out.println("[LOG] Main program finished.");
		
		// Syncronize result
		System.out.println("[LOG] Result size = " + result.size());
		long endTime = System.currentTimeMillis();
		
		// Uncomment the following lines to see the result keys
			//TreeMap<String, String> tm = result.getValue();
			//while (!tm.isEmpty()) { System.out.println(tm.pollFirstEntry().getKey());}
		
		System.out.println("[TIMER] Elapsed time: " + (endTime - startTime) + " ms");
	}
	
	private void run() {
		// Initialize file Names
		System.out.println("[LOG] Initialising filenames for each matrix");
		initializeVariables();
		
		// Compute result
		System.out.println("[LOG] Computing result");
		for (int i = 0; i < filePaths.length; ++i) {
			partialSorted[i] = (new SortImpl()).sortPartitionFromFile(filePaths[i]);
		}

		result = mergeReduce(partialSorted);		
	}

	private void initializeVariables () {		
		NUM_FRAGMENTS = new File(DATA_FOLDER).listFiles().length;
		filePaths = new String[NUM_FRAGMENTS];
		int i = 0;
		for (File f : new File(DATA_FOLDER).listFiles()) {
			filePaths[i] = f.getAbsolutePath();
			i = i + 1;
		}
		System.out.println("NUM_FRAGMENTS value = " + NUM_FRAGMENTS);
		
		partialSorted = new SortedTreeMap[NUM_FRAGMENTS];
		result = new SortedTreeMap();
	}
	
	private SortedTreeMap mergeReduce(SortedTreeMap[] data) {
		LinkedList<SortedTreeMap> q = new LinkedList<SortedTreeMap>(Arrays.asList(data));
		while (!q.isEmpty()) {
			SortedTreeMap m1 = q.poll();
			if (!q.isEmpty()) {
				SortedTreeMap m2 = q.poll();
				SortedTreeMap m3 = (new SortImpl()).reduceTask(m1, m2);
				q.offer(m3);
			} else {
				return m1;
			}
		}
		
		// Should never reach this point
		return null;
	}

}
