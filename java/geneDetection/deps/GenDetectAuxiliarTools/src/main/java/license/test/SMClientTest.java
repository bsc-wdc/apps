package license.test;

import eu.optimis.service_manager.client.ServiceManagerClient;

public class SMClientTest {
	public static void main(String[] args) {
        try {
        	
        	ServiceManagerClient sm_client = new ServiceManagerClient("optimis-spvm.atosorigin.es","8080");
            
        } catch (Exception e) {
            e.printStackTrace();
        }
	}
}
