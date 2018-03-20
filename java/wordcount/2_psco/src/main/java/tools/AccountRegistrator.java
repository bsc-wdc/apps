package tools;

import client.ClientManagementLib;
import util.ids.AccountID;
import util.management.accountmgr.Account;
import util.management.accountmgr.AccountRole;
import util.management.accountmgr.Credential;
import util.management.accountmgr.PasswordCredential;


public class AccountRegistrator {

	public static void main(String[] args) throws Exception {
		if (args.length != 4) {
			System.err.println("\n Bad arguments. Usage: \n\n " + AccountRegistrator.class.getName()
					+ " <admin_name> <admin_pass> <newaccount_name> <newaccount_pass> \n");
			return;
		}
		String adminName = args[0];
		String adminPass = args[1];
		String newAccName = args[2];
		String newAccPass = args[3];

		// Get accounts info
		AccountID adminID = ClientManagementLib.getAccountID(adminName);
		Credential adminPassCred = new PasswordCredential(adminPass);
		Credential newAccPassCred = new PasswordCredential(newAccPass);
		Account newAccount = new Account(newAccName, AccountRole.NORMAL_ROLE, newAccPassCred);
		AccountID newAccID = ClientManagementLib.newAccount(adminID, adminPassCred, newAccount);

		if (newAccID != null) {
			System.out.println("Account created for user " + newAccName + " with ID " + newAccID);
		} else {
			System.out.println("Account " + newAccName + " could not be created.");
		}
	}
}
