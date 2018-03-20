package sortByKey.random;

import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.TreeMap;


public class Sort {
	private static int NUM_KEYS;
	private static int UNIQUE_KEYS;
	private static int KEY_LENGTH;
	private static int UNIQUE_VALUES;
	private static int VALUE_LENGTH;
	private static int NUM_FRAGMENTS;
	private static int RANDOM_SEED;
	private static int KEYS_PER_FRAGMENT;
	
	private static SortedTreeMap[] partialSorted;
	private static SortedTreeMap result;
	
	public static void main(String args[]) {
		//Get parameters
		if (args.length != 7) {
			System.out.println("[ERROR] Usage: Sort <NUM_KEYS> <UNIQUE_KEYS> <KEY_LENGTH>");
			System.out.println("                    <UNIQUE_VALUES> <VALUE_LENGTH> <NUM_FRAGMENTS> <RANDOM_SEED>");
			System.exit(-1);
		}
		NUM_KEYS = Integer.valueOf(args[0]);
		UNIQUE_KEYS = Integer.valueOf(args[1]);
		KEY_LENGTH = Integer.valueOf(args[2]);
		UNIQUE_VALUES = Integer.valueOf(args[3]);
		VALUE_LENGTH = Integer.valueOf(args[4]);
		NUM_FRAGMENTS = Integer.valueOf(args[5]);
		RANDOM_SEED = Integer.valueOf(args[6]);
		KEYS_PER_FRAGMENT = NUM_KEYS/NUM_FRAGMENTS;
		
		System.out.println("NUM_KEYS parameter value = " + NUM_KEYS);
		System.out.println("UNIQUE_KEYS parameter value = " + UNIQUE_KEYS);
		System.out.println("KEY_LENGTH parameter value = " + KEY_LENGTH);
		System.out.println("UNIQUE_VALUES parameter value = " + UNIQUE_VALUES);
		System.out.println("VALUE_LENGTH parameter value = " + VALUE_LENGTH);
		System.out.println("NUM_FRAGMENTS parameter value = " + NUM_FRAGMENTS);
		System.out.println("RANDOM_SEED parameter value = " + RANDOM_SEED);
		System.out.println("KEYS_PER_FRAGMENT parameter value = " + KEYS_PER_FRAGMENT);
		
		// Run sort by key app
		long startTime = System.currentTimeMillis();
		Sort sbk = new Sort();
		sbk.run();
		
		System.out.println("[LOG] Main program finished.");
		
		// Syncronize result
		System.out.println("[LOG] Result size = " + result.size());
		long endTime = System.currentTimeMillis();
		
		// Uncomment the following lines to see the result keys
		TreeMap<String, String> tm = result.getValue();
		while (!tm.isEmpty()) { System.out.println(tm.pollFirstEntry().getKey());}
		
		System.out.println("[TIMER] Elapsed time: " + (endTime - startTime) + " ms");
	}
	
	private void run() {
		// Initialize file Names
		System.out.println("[LOG] Initialising filenames for each matrix");
		initializeVariables();
		
		// Compute result
		System.out.println("[LOG] Computing result");
		for (int i = 0; i < NUM_FRAGMENTS; ++i) {
			HashMap<String,String> fragment = (new SortImpl()).generateFragment(KEYS_PER_FRAGMENT, UNIQUE_KEYS, KEY_LENGTH, UNIQUE_VALUES, VALUE_LENGTH, RANDOM_SEED + (long)2*KEYS_PER_FRAGMENT*i);
			partialSorted[i] = (new SortImpl()).sortPartition(fragment);
		}

		result = mergeReduce(partialSorted);		
	}

	private void initializeVariables () {	
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
