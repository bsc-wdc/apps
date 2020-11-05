package randomforest.client;

import es.bsc.compss.agent.rest.types.messages.StartApplicationRequest;

import javax.ws.rs.client.Client;
import javax.ws.rs.client.ClientBuilder;
import javax.ws.rs.client.Entity;
import javax.ws.rs.client.WebTarget;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;

import org.glassfish.jersey.client.ClientConfig;


public class DatasetGenerator {

    private static final String MASTER = "127.0.0.1";
    private static final int MASTER_PORT = 46_101;

    private static final ClientConfig CONFIG = new ClientConfig();
    private static final Client CLIENT = ClientBuilder.newClient(CONFIG);
    private static final WebTarget TARGET = CLIENT.target("http://" + MASTER + ":" + MASTER_PORT);


    public static void main(String[] args) throws Exception {
        String className = "randomforest.RandomForest";
        String methodName = "createDataSet";

        StartApplicationRequest sar = new StartApplicationRequest();
        sar.setClassName(className);

        int numSamples = 20;
        int numFeatures = 10;
        int numClasses = 7;
        int numInformative = 4;
        int numRedundant = 2;
        int numRepeated = 1;
        int numClustersPerClass = 2;
        boolean shuffle = true;
        Long randomSeed = 0L;

        sar.setMethodName(methodName);
        sar.addParameter(numSamples);
        sar.addParameter(numFeatures);
        sar.addParameter(numClasses);
        sar.addParameter(numInformative);
        sar.addParameter(numRedundant);
        sar.addParameter(numRepeated);
        sar.addParameter(numClustersPerClass);
        sar.addParameter(shuffle);
        sar.addParameter(randomSeed);

        WebTarget wt = TARGET.path("/COMPSs/startApplication/");
        Response response = wt.request(MediaType.APPLICATION_JSON).put(Entity.xml(sar), Response.class);

        System.out.println(response.getStatusInfo().getStatusCode());
        if (response.getStatusInfo().getStatusCode() != 200) {
            System.out.println("Could not request application " + "randomforest.RandomForest" + " end to " + wt);
        }
        System.out.println(response.readEntity(String.class));

    }

}
