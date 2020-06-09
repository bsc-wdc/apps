
package randomforest.client;

import es.bsc.compss.agent.rest.types.messages.StartApplicationRequest;
import javax.ws.rs.client.Client;
import javax.ws.rs.client.ClientBuilder;
import javax.ws.rs.client.Entity;
import javax.ws.rs.client.WebTarget;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import org.glassfish.jersey.client.ClientConfig;


public class Main {

    private static final String MASTER = "127.0.0.1";
    private static final int MASTER_PORT = 46101;

    private static final ClientConfig config = new ClientConfig();
    private static final Client client = ClientBuilder.newClient(config);
    private static final WebTarget target = client.target("http://" + MASTER + ":" + MASTER_PORT);


    public static void main(String[] args) throws Exception {
        String className = "randomforest.RandomForest";
        String methodName = "generateRandomModel";

        StartApplicationRequest sar = new StartApplicationRequest();
        sar.setClassName(className);
        sar.setMethodName(methodName);
        sar.setCeiClass(className + "Itf");

        String numSamples = 30_000 + "";
        String numFeatures = 40 + "";
        // String numClasses = 200 + "";
        // String numInformative = 20 + "";
        // String numRedundant = 2 + "";
        // String numRepeated = 1 + "";
        // String numClustersPerClass = 2 + "";
        // String shuffle = true + "";
        // String randomSeed = "0";
        // String numEstimators = "12";

        sar.addParameter(numSamples);
        sar.addParameter(numFeatures);
        // sar.addParameter(new String[]{numSamples, numFeatures, numClasses, numInformative, numRedundant, numRepeated,
        // numClustersPerClass, shuffle, randomSeed, numEstimators});
        Debugger.debugAsXML(sar);
        WebTarget wt = target.path("/COMPSs/startApplication/");
        Response response = wt.request(MediaType.APPLICATION_JSON).put(Entity.xml(sar), Response.class);

        System.out.println(response.getStatusInfo().getStatusCode());
        if (response.getStatusInfo().getStatusCode() != 200) {
            System.out.println("Could not request application " + "randomforest.RandomForest" + " end to " + wt);
        }
        System.out.println(response.readEntity(String.class));

    }

}
