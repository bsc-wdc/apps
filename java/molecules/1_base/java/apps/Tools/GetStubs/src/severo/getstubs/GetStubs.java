package severo.getstubs;

import java.util.*;
import java.util.Map.Entry;

import util.credentials.*;
import util.ids.*;
import util.management.contractmgr.*;
import client.ClientManagementLib;

public class GetStubs {

    public static void main(String[] args) throws Exception {

        if (args.length != 3) {
            System.err.println("Bad arguments, expected: <user> <password> <targetdir>");
            System.exit(-1);
        }

        AccountID accountID = ClientManagementLib.getAccountID(args[0]);
        if (accountID == null) {
            System.err.println("AccountID of " + args[0] + " cannot be retrieved.");
            System.exit(-1);
        }

        PasswordCredential password = new PasswordCredential(args[1]);

	Map<ContractID, Contract> accountContracts = 
            ClientManagementLib.getContractsOfApplicant(accountID, password);

        LinkedList<ContractID> contractsIDs = new LinkedList<ContractID>(accountContracts.keySet());
        ClientManagementLib.getAndStoreStubs(accountID, password, contractsIDs, args[2]);

	ClientManagementLib.finishConnections();
    }	
}