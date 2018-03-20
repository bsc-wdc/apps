package tools;

import java.util.Calendar;
import java.util.HashSet;
import java.util.Set;

import client.ClientManagementLib;
import util.ids.AccountID;
import util.ids.DataContractID;
import util.ids.DataSetID;
import util.management.accountmgr.Credential;
import util.management.accountmgr.PasswordCredential;
import util.management.datacontractmgr.DataContract;
import util.management.datasetmgr.DataSet;


public class DataContractRegistrator {

	public static void main(String[] args) throws Exception {
		if (args.length != 4) {
			System.err.println("\n Bad arguments. Usage: \n\n " + DataContractRegistrator.class.getName()
					+ " <dataset_name> <owner_name> <owner_pass> <benef_name> \n");
			return;
		}
		String datasetName = args[0];
		String ownerName = args[1];
		String ownerPass = args[2];
		String benefName = args[3];

		AccountID ownerID = ClientManagementLib.getAccountID(ownerName);
		Credential ownerCredential = new PasswordCredential(ownerPass);
		DataSetID datasetID = ClientManagementLib.getDatasetID(ownerID, ownerCredential, datasetName);
		if (datasetID == null) {
			DataSet newDataSet = new DataSet(datasetName, ownerName);
			datasetID = ClientManagementLib.newDataSet(ownerID, ownerCredential, newDataSet);
		}
		System.out.println(" == Dataset " + datasetName + " with ID " + datasetID);

		Set<String> applicants = new HashSet<String>();
		applicants.add(benefName);
		Calendar beginDate = Calendar.getInstance();
		beginDate.add(Calendar.YEAR, -1);
		Calendar endDate = Calendar.getInstance();
		endDate.add(Calendar.YEAR, 1);
		DataContract dContract = new DataContract(datasetName, ownerName, applicants, beginDate, endDate);

		// Create a new contract for the benef to access it
		DataContractID dContractID = ClientManagementLib.newPrivateDataContract(ownerID, ownerCredential, dContract);
		if (dContractID != null) {
			System.out.println(" == Created data contract " + dContractID);
		} else {
			System.out.println(" == Data contract was not created");
		}

		ClientManagementLib.finishConnections();
	}
}
