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

package simple;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;


public class Simple {
		
	public static void main(String[] args) {
		String counterName = "counter";
		int initialValue = Integer.parseInt(args[0]);	

		// Initialize counter (c -> 1)
		try {
			FileOutputStream fos = new FileOutputStream(counterName);
			fos.write(initialValue);
			System.out.println("Initial counter value is " + initialValue);
			fos.close();
		}
		catch(IOException ioe) {
			ioe.printStackTrace();
		}

		
		// Execute increment (c -> 2)
		for (int i = 0; i < initialValue; i++) {
            SimpleImpl.increment(counterName);
        }


		// Open the file and print final counter value (should be 2)
		try {
        	FileInputStream fis = new FileInputStream(counterName);
            System.out.println("Final counter value is " + fis.read());
			fis.close();
		}
		catch(IOException ioe) {
			ioe.printStackTrace();
		}
	}

}
