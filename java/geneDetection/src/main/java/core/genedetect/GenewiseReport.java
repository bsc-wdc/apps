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

package core.genedetect;

import java.io.Serializable;

import core.genedetect.GFF;
import core.genedetect.NemusString;

public class GenewiseReport implements Serializable {
	/**
	 * Serial version out of Runtime cannot be 1L nor 2L
	 */
	private static final long serialVersionUID = 3L;

	private GFF gff;
	
	public GenewiseReport() {
	}
	
	public GenewiseReport(GFF gff) {
		this.gff = gff;
	}
	
	public GFF getGff() {
		return gff;
	}

	public void setGff(GFF gff) {
		this.gff = gff;
	}

	public void mergeGenewiseResults(GenewiseReport report) {
		String mergedContent = gff.getContent().getValue() + "\n" + report.getGff().getContent().getValue();
    	NemusString newContent = new NemusString();
    	newContent.setValue(mergedContent);
    	gff.setContent(newContent);
	}
	
}
