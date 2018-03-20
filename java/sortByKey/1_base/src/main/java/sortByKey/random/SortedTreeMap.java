package sortByKey.random;

import java.io.Serializable;
import java.util.Comparator;
import java.util.HashMap;
import java.util.TreeMap;


public class SortedTreeMap implements Serializable {
	
	private TreeMap<String,String> value;
	
	public SortedTreeMap() {
		this.value = new TreeMap<String,String>(new SerializableComparator());
	}
	
	public SortedTreeMap(SortedTreeMap stm) {
		this.value = new TreeMap<String,String>(new SerializableComparator());
		this.value.putAll(stm.getValue());
	}
	
	public SortedTreeMap(HashMap<String, String> map) {
		this.value = new TreeMap<String,String>(new SerializableComparator());
		this.value.putAll(map);
	}
	
	public TreeMap<String, String> getValue() {
		return this.value;
	}
	
	public void setValue(TreeMap<String, String> val) {
		this.value = new TreeMap<String,String>(new SerializableComparator());
		this.value.putAll(val);
	}
	
	public void insert(String key, String value) {
		this.value.put(key, value);
	}
	
	public void insertAll(SortedTreeMap stm) {
		this.value.putAll(stm.getValue());
	}
	
	public void remove(String key) {
		this.value.remove(key);
	}
	
	public int size() {
		return this.value.size();
	}
	
	public class SerializableComparator implements Serializable, Comparator<String> {
		@Override
        public int compare(String o1, String o2) {
            return o1.compareTo(o2);
		}
	}
}
