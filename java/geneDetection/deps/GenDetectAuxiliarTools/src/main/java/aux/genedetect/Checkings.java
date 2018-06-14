package aux.genedetect;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.net.URL;
import java.net.URLConnection;

public class Checkings {
	
	public static void checkConnectivity() {
    	System.out.println("Checking Access to the service");
    	try{ 
    		URL url = new URL("http://bscgrid05.bsc.es:20390/biotools/biotools?wsdl");
    		URLConnection conn = url.openConnection();
    		BufferedReader rd = new BufferedReader(new InputStreamReader(conn.getInputStream()));
    		StringBuffer sb = new StringBuffer();
    		String line;
    		while ((line = rd.readLine()) != null)
    		{
    			sb.append(line);
    		}
    		rd.close();
    		if (sb.toString().length()>0){
    			System.out.println("Connection Possible");
    		}else{
    			System.out.println("Connection to service NOT Possible");
    		}
    	} catch (Exception e){
    		System.out.println("Connection to service NOT Possible");
    		e.printStackTrace();
    	}
		
	}

	public static boolean checkLicense() {
		File f = new File("/mnt/context/licensetoken/license.token.0");
		return f.exists();
	}
	
	
}
