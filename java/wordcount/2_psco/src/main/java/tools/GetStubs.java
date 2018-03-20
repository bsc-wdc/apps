package tools;

import java.io.File;
import java.util.LinkedList;
import java.util.Map;

import client.ClientManagementLib;
import util.FileAndAspectsUtils;
import util.ids.AccountID;
import util.ids.ContractID;
import util.management.accountmgr.Credential;
import util.management.accountmgr.PasswordCredential;
import util.management.contractmgr.Contract;


public class GetStubs {

	public static void main(String[] args) throws Exception {
		if (args.length != 3) {
			System.err.println("\n Bad arguments. Usage: \n\n " + GetStubs.class.getName() + " <user_name> <user_pass> <stubs_path> \n");
			return;
		}
		String userName = args[0];
		String userPass = args[1];
		String userStubsPath = args[2];

		// Get account info
		AccountID userID = ClientManagementLib.getAccountID(userName);
		Credential userCredential = new PasswordCredential(userPass);

		// Clean directory of consumer
		FileAndAspectsUtils.deleteFolderContent(new File(userStubsPath));

		// Contracts with all the domains
		System.out.println("Getting user's contracts.");
		LinkedList<ContractID> contractsIDs = new LinkedList<ContractID>();
		Map<ContractID, Contract> contracts = ClientManagementLib.getContractsOfApplicant(userID, userCredential);
		contractsIDs.addAll(contracts.keySet());
		System.out.println("Contracts obtained " + contractsIDs);

		System.out.println("Getting stubs and storing them.");
		ClientManagementLib.getAndStoreStubs(userID, userCredential, contractsIDs, userStubsPath);
		System.out.println("Stubs stored in " + userStubsPath);
	}
}
