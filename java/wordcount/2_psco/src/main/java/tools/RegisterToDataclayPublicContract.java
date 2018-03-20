package tools;

import client.ClientManagementLib;
import util.ids.AccountID;
import util.ids.ContractID;
import util.management.accountmgr.Credential;
import util.management.accountmgr.PasswordCredential;


public class RegisterToDataclayPublicContract {

	public static void main(String[] args) throws Exception {
		if (args.length != 2) {
			System.err.println("\n Bad arguments. Usage: \n\n " + RegisterToDataclayPublicContract.class.getName()
					+ " <applicant_username> <applicant_pass> \n");
			return;
		}
		String applicantName = args[0];
		String applicantPass = args[1];

		AccountID applicantID = ClientManagementLib.getAccountID(applicantName);
		Credential applicantCredential = new PasswordCredential(applicantPass);
		ContractID publicContract = ClientManagementLib.getContractOfDataClayProvider(applicantID, applicantCredential);
		boolean result = ClientManagementLib.registerToPublicContract(applicantID, applicantCredential, publicContract);
		if (result) {
			System.out.println("Account " + applicantName + "[" + applicantID
					+ "] has been correctly registered to public dataClay contract");
		} else {
			System.err.println("ERROR. Account " + applicantName + "[" + applicantID
					+ "] could not be registered to public dataClay contract");
		}

		System.out.println("Finish connections");
		ClientManagementLib.finishConnections();
	}
}
