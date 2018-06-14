
package core.genedetect;

import java.io.Serializable;

import core.genedetect.GFF;
import core.genedetect.NemusString;

public class GenewiseReport implements Serializable {

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
