package es.bsc.genedetection.gui;

public class ServiceCaller {

	public static String call(String bd, String seq, int alignment, int scores, float threshold, String tempId) throws Exception {
		es.bsc.genedetection.GeneDetectionService serv = new es.bsc.genedetection.GeneDetectionService(	new java.net.URL("http://localhost:20341/GeneDetection/GeneDetection"),new javax.xml.namespace.QName("http://genedetection.bsc.es/","GeneDetectionService"));
		es.bsc.genedetection.GeneDetection port = serv.getGeneDetectionPort();
		String result = port.detectGenes(bd, seq, alignment, scores, threshold);
		//String result = getDefaultResult();
		String tempFile = "/tmp/" + tempId;
		java.io.FileWriter fw = new java.io.FileWriter(tempFile);
		java.io.BufferedWriter bw = new java.io.BufferedWriter(fw);
		bw.write(result);
		bw.close();
		fw.close();
		String[] cmd = {
				"/bin/sh",
				"-c",
				"/GFF2PS/gff2ps_v0.98d "
				+ tempFile
				+ " | gs -sDEVICE=jpeg -sOutputFile="+System.getProperty("catalina.base")+"/webapps/GeneDetectionGUI/images/result-"
				+ tempId + ".jpg -r300"};
		System.out.println(cmd[2]);
		Process p = Runtime.getRuntime().exec(cmd);
		p.waitFor();
		cmd[2] = "convert -rotate 90 "+System.getProperty("catalina.base")+"/webapps/GeneDetectionGUI/images/result-"
				+ tempId + ".jpg "+System.getProperty("catalina.base")+"/webapps/GeneDetectionGUI/images/result-"
				+ tempId + ".jpg";
		System.out.println(cmd[2]);
		p = Runtime.getRuntime().exec(cmd);
		p.waitFor();
		return result;

	}

	private static String getDefaultResult() {
		return "ENSEMBL|chr4	GeneWise	match	24111043	24111123	27.28	+	.	ENSEMBL|chr4-genewise-prediction-1\n"
				+ "\n"
				+ "\n"
				+ "ENSEMBL|chr4	GeneWise	cds	24111043	24111123	0.00	+	0	ENSEMBL|chr4-genewise-prediction-1\n"
				+ "\n"
				+ "\n"
				+ "ENSEMBL|chr4	GeneWise	match	22881811	22881861	21.20	+	.	ENSEMBL|chr4-genewise-prediction-1\n"
				+ "\n"
				+ "ENSEMBL|chr4	GeneWise	cds	22881811	22881861	0.00	+	0	ENSEMBL|chr4-genewise-prediction-1\n"
				+ "\n"
				+ "\n"
				+ "ENSEMBL|chr4	GeneWise	match	8781400	8781519	22.30	+	.	ENSEMBL|chr4-genewise-prediction-1\n"
				+ "\n"
				+ "ENSEMBL|chr4	GeneWise	cds	8781400	8781519	0.00	+	0	ENSEMBL|chr4-genewise-prediction-1\n"
				+ "\n"
				+ "\n"
				+ "ENSEMBL|chr4	GeneWise	match	11089344	11089382	21.55	-	.	ENSEMBL|chr4-genewise-prediction-1\n"
				+ "\n"
				+ "ENSEMBL|chr4	GeneWise	cds	11089344	11089382	0.00	-	0	ENSEMBL|chr4-genewise-prediction-1\n"
				+ "\n"
				+ "\n"
				+ "ENSEMBL|chr4	GeneWise	match	6336897	6336941	25.06	+	.	ENSEMBL|chr4-genewise-prediction-1\n"
				+ "\n"
				+ "ENSEMBL|chr4	GeneWise	cds	6336897	6336941	0.00	+	0	ENSEMBL|chr4-genewise-prediction-1\n"
				+ "\n"
				+ "\n"
				+ "ENSEMBL|chr4	GeneWise	match	3951196	3951249	21.61	-	.	ENSEMBL|chr4-genewise-prediction-1\n"
				+ "\n"
				+ "ENSEMBL|chr4	GeneWise	cds	3951196	3951249	0.00	-	0	ENSEMBL|chr4-genewise-prediction-1\n"
				+ "\n"
				+ "\n"
				+ "ENSEMBL|chr4	GeneWise	match	3076100	3076144	22.53	-	.	ENSEMBL|chr4-genewise-prediction-1\n"
				+ "\n"
				+ "ENSEMBL|chr4	GeneWise	cds	3076100	3076144	0.00	-	0	ENSEMBL|chr4-genewise-prediction-1\n"
				+ "\n"
				+ "\n"
				+ "ENSEMBL|chr4	GeneWise	match	6154017	6154052	19.42	-	.	ENSEMBL|chr4-genewise-prediction-1\n"
				+ "\n"
				+ "ENSEMBL|chr4	GeneWise	cds	6154017	6154052	0.00	-	0	ENSEMBL|chr4-genewise-prediction-1\n"
				+ "\n"
				+ "\n"
				+ "ENSEMBL|chr4	GeneWise	match	21565422	21565502	25.74	-	.	ENSEMBL|chr4-genewise-prediction-1\n"
				+ "\n"
				+ "ENSEMBL|chr4	GeneWise	cds	21565422	21565502	0.00	-	0	ENSEMBL|chr4-genewise-prediction-1\n"
				+ "\n"
				+ "\n"
				+ "ENSEMBL|chr4	GeneWise	match	3951196	3951249	21.61	-	.	ENSEMBL|chr4-genewise-prediction-1\n"
				+ "\n"
				+ "ENSEMBL|chr4	GeneWise	cds	3951196	3951249	0.00	-	0	ENSEMBL|chr4-genewise-prediction-1\n"
				+ "\n"
				+ "\n"
				+ "ENSEMBL|chr4	GeneWise	match	11089344	11089382	21.55	-	.	ENSEMBL|chr4-genewise-prediction-1\n"
				+ "\n"
				+ "ENSEMBL|chr4	GeneWise	cds	11089344	11089382	0.00	-	0	ENSEMBL|chr4-genewise-prediction-1\n"
				+ "\n"
				+ "\n"
				+ "\n"
				+ "ENSEMBL|chr4	GeneWise	match	15669658	15669708	25.49	+	.	ENSEMBL|chr4-genewise-prediction-1\n"
				+ "\n"
				+ "ENSEMBL|chr4	GeneWise	cds	15669658	15669708	0.00	+	0	ENSEMBL|chr4-genewise-prediction-1\n"
				+ "\n"
				+ "\n"
				+ "ENSEMBL|chr4	GeneWise	match	11089344	11089382	21.55	-	.	ENSEMBL|chr4-genewise-prediction-1\n"
				+ "\n"
				+ "ENSEMBL|chr4	GeneWise	cds	11089344	11089382	0.00	-	0	ENSEMBL|chr4-genewise-prediction-1\n"
				+ "\n"
				+ "\n"
				+ "ENSEMBL|chr4	GeneWise	match	21385216	21385299	27.94	-	.	ENSEMBL|chr4-genewise-prediction-1\n"
				+ "\n"
				+ "ENSEMBL|chr4	GeneWise	cds	21385216	21385299	0.00	-	0	ENSEMBL|chr4-genewise-prediction-1\n"
				+ "\n"
				+ "\n"
				+ "ENSEMBL|chr4	GeneWise	match	3951196	3951249	21.61	-	.	ENSEMBL|chr4-genewise-prediction-1\n"
				+ "\n"
				+ "ENSEMBL|chr4	GeneWise	cds	3951196	3951249	0.00	-	0	ENSEMBL|chr4-genewise-prediction-1\n"
				+ "\n"
				+ "\n"
				+ "ENSEMBL|chr4	GeneWise	match	4947621	4947656	20.25	-	.	ENSEMBL|chr4-genewise-prediction-1\n"
				+ "\n"
				+ "ENSEMBL|chr4	GeneWise	cds	4947621	4947656	0.00	-	0	ENSEMBL|chr4-genewise-prediction-1\n"
				+ "\n"
				+ "\n"
				+ "ENSEMBL|chr4	GeneWise	match	11089344	11089382	21.55	-	.	ENSEMBL|chr4-genewise-prediction-1\n"
				+ "\n"
				+ "ENSEMBL|chr4	GeneWise	cds	11089344	11089382	0.00	-	0	ENSEMBL|chr4-genewise-prediction-1\n"
				+ "\n"
				+ "\n"
				+ "ENSEMBL|chr4	GeneWise	match	18786277	18786327	24.21	+	.	ENSEMBL|chr4-genewise-prediction-1\n"
				+ "\n"
				+ "ENSEMBL|chr4	GeneWise	cds	18786277	18786327	0.00	+	0	ENSEMBL|chr4-genewise-prediction-1\n"
				+ "\n"
				+ "\n"
				+ "ENSEMBL|chr4	GeneWise	match	8550511	8550585	23.01	+	.	ENSEMBL|chr4-genewise-prediction-1\n"
				+ "\n"
				+ "ENSEMBL|chr4	GeneWise	cds	8550511	8550585	0.00	+	0	ENSEMBL|chr4-genewise-prediction-1\n"
				+ "\n"
				+ "\n"
				+ "ENSEMBL|chr4	GeneWise	match	18119129	18119170	23.12	-	.	ENSEMBL|chr4-genewise-prediction-1\n"
				+ "\n"
				+ "ENSEMBL|chr4	GeneWise	cds	18119129	18119170	0.00	-	0	ENSEMBL|chr4-genewise-prediction-1\n"
				+ "\n"
				+ "\n"
				+ "ENSEMBL|chr4	GeneWise	match	24111043	24111123	26.50	+	.	ENSEMBL|chr4-genewise-prediction-1\n"
				+ "\n"
				+ "ENSEMBL|chr4	GeneWise	cds	24111043	24111123	0.00	+	0	ENSEMBL|chr4-genewise-prediction-1\n"
				+ "\n";
	}
}
