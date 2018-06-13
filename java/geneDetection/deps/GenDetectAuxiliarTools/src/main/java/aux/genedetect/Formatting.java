package aux.genedetect;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.InputStreamReader;
import java.io.Reader;
import core.genedetect.Database;
import core.genedetect.FASTA;
import core.genedetect.Filter;
import core.genedetect.Matrix;
import core.genedetect.NemusString;
import core.genedetect.Program;
import core.genedetect.RunBlast2GeneSecondaryParameters;
import core.genedetect.RunNCBIBlastAgainstDBFromFASTASecondaryParameters;
import core.genedetect.RunNCBIBlastpSecondaryParameters;

public class Formatting {
	
	private static final String SERVICE_DIR = "/optimis_service/";
	private static final String NAMESPACE = "ENSEMBL";;
	
	public static FASTA loadSequenceFromFile(String sequence) {
        StringBuilder writer = new StringBuilder();
        char[] buffer = new char[1024];

        try {
            InputStreamReader is = new FileReader(SERVICE_DIR + "sequence/" + sequence + ".fasta");
            Reader reader = new BufferedReader(is);
            int n;
            while ((n = reader.read(buffer)) != -1) {
                writer.append(buffer, 0, n);
            }
            is.close();
        } catch (Exception e) {
            System.err.println("Error loading sequence from file");
            System.exit(1);
        }

        String fStr = writer.toString();
        NemusString content = new NemusString();
        content.setValue(fStr);

        FASTA fasta = new FASTA();
        fasta.setNemusId(sequence);
        fasta.setNemusNamespace(NAMESPACE);
        fasta.setContent(content);

        return fasta;
    }
    
	public static RunNCBIBlastpSecondaryParameters generateNCBIBlastpParameters(int alignaments, int scores, float threshold) {
        RunNCBIBlastpSecondaryParameters params = new RunNCBIBlastpSecondaryParameters();

        params.setAlignments(alignaments);
        params.setScores(scores);
        params.setExpectedThreshold(threshold);
        //params.setExpectedThreshold(1e-10f);
        params.setDatabase(Database.UNI_PROT);
        params.setFilter(Filter.FALSE);

        return params;
    }

    public static RunNCBIBlastAgainstDBFromFASTASecondaryParameters generateNCBIBlastParameters(float threshold) {
        RunNCBIBlastAgainstDBFromFASTASecondaryParameters params = new RunNCBIBlastAgainstDBFromFASTASecondaryParameters();

        params.setProgram(Program.TBLASTN);
        params.setMatrix(Matrix.PAM_70);
        params.setExtendgap(1);
        params.setOpengap(10);
        params.setFilter(Filter.FALSE);
        //params.setExpectedThreshold(threshold);

        return params;
    }

    public static RunBlast2GeneSecondaryParameters generateBlast2GeneParameters() {
        RunBlast2GeneSecondaryParameters params = new RunBlast2GeneSecondaryParameters();

        params.setCoverage(0.7f);

        return params;
    }

}
