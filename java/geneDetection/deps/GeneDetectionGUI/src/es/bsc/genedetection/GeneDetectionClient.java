package es.bsc.genedetection;

import java.net.MalformedURLException;
import java.net.URL;

import javax.xml.namespace.QName;

public class GeneDetectionClient {

	public static void main(String[] args) throws MalformedURLException {
		GeneDetectionService serv = new GeneDetectionService(new URL("http://130.239.48.11:8080/GeneDetection/GeneDetection"), new QName("http://genedetection.bsc.es/", "GeneDetectionService"));
		GeneDetection port = serv.getGeneDetectionPort();
		System.out.println(port.detectGenes("chr4", "ENSRNOP00000053114", 6, 6, 1e-10f));
	}

}
