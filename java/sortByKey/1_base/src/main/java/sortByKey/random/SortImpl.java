package sortByKey.random;

import java.util.HashMap;
import java.util.Random;


public class SortImpl {

	public SortedTreeMap reduceTask(SortedTreeMap m1, SortedTreeMap m2) {
		m2.insertAll(m1);
		return m2;
	}
	
	public HashMap<String, String> generateFragment(int numKeys, int uniqueKeys, int keyLength, int uniqueValues,
			int valueLength, long randomSeed) {
		HashMap<String, String> res = new HashMap<String, String>();
		Random generator = new Random(randomSeed);
		for (int i = 0; i < numKeys; ++i) {
			int key = (int)(generator.nextDouble()*uniqueKeys);
			int value = (int)(generator.nextDouble()*uniqueValues);
			String skey = String.valueOf(key);
			skey = String.format("%" + keyLength + "s", skey).replace(' ', '0');
			String svalue = String.valueOf(value);
			svalue = String.format("%" + valueLength + "s", svalue).replace(' ', '0');
			res.put(skey,  svalue);
		}
		
		return res;
	}
	
	public SortedTreeMap sortPartition(HashMap<String,String> fragment) {
		SortedTreeMap res = new SortedTreeMap(fragment);
		return res;
	}

}
