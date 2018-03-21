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
package molecule;

import java.io.Serializable;

public class SumMolecules implements Serializable {

	public static int NMOLECULES = 10;

	public static void main(String[] args) {
				
		int n = NMOLECULES;
		if (args.length != 0) {
		   n = Integer.parseInt(args[0]);
		}
		sum(1, n);		
		
	}

	public static COMPSsMolecule sum(int start, int end) {

		COMPSsMolecule a = null;
		COMPSsMolecule b = null;

		if (start == end) {			
			a = COMPSsMolecule.getMolecule(start);
			return a;
		} else {
			if ((end - start) == 1) {
				a = COMPSsMolecule.getMolecule(start);
				b = COMPSsMolecule.getMolecule(end);
				a.addMolecule(b);
				return a;
			} else {
				int startA = start;
				int endA = ((end - start) / 2) + start;
				int startB = endA + 1;
				int endB = end;												
				a = sum(startA, endA);
				b = sum(startB, endB);
				a.addMolecule(b);
				return a;
			}
		}
	}			
}
