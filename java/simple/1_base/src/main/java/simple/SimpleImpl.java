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
import java.io.FileNotFoundException;


public class SimpleImpl {

	public static void increment(String counterFile) {
		try
		{
			FileInputStream fis = new FileInputStream(counterFile);
			int count = fis.read();
			fis.close();

			FileOutputStream fos = new FileOutputStream(counterFile);
            fos.write(++count);			
			fos.close();
		}
		catch(FileNotFoundException fnfe)
		{
			fnfe.printStackTrace();
		}
		catch(IOException ioe)
		{
			ioe.printStackTrace();
		}
	}
	
}
