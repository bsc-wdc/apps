package sortByKey.files;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;


public class SortImpl {
	
	public SortedTreeMap sortPartitionFromFile(String filePath) {
		File file = new File(filePath);
		SortedTreeMap res  = new SortedTreeMap();
		
		FileReader fr = null;
		BufferedReader br = null;
		try {
			fr = new FileReader(file);
			br = new BufferedReader(fr);
			String line;
			while ((line = br.readLine()) != null) {
				String[] keyValuePair = line.split(" ");
				res.insert(keyValuePair[0], keyValuePair[1]);
			}
		} catch (Exception e) {
			System.err.println("ERROR: Cannot retrieve values from " + file.getName());
			e.printStackTrace();
		} finally {
			if (br != null) {
				try {
					br.close();
				} catch (Exception e) {
					System.err.println("ERROR: Cannot close buffered reader on file " + file.getName());
					e.printStackTrace();
				}
			}
			if (fr != null) {
				try {
					fr.close();
				} catch (Exception e) {
					System.err.println("ERROR: Cannot close file reader on file " + file.getName());
					e.printStackTrace();
				}
			}
		}
		
		return res;
	}
	
	public SortedTreeMap reduceTask(SortedTreeMap m1, SortedTreeMap m2) {
		SortedTreeMap res = new SortedTreeMap(m1);
		res.insertAll(m2);

		return res;
	}

}
