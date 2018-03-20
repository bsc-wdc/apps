package tools;

import java.util.Calendar;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import client.ClientManagementLib;
import util.ids.AccountID;
import util.ids.ContractID;
import util.ids.DomainID;
import util.ids.InterfaceID;
import util.ids.MetaClassID;
import util.management.accountmgr.Credential;
import util.management.accountmgr.PasswordCredential;
import util.management.classmgr.MetaClass;
import util.management.classmgr.Operation;
import util.management.classmgr.Property;
import util.management.contractmgr.Contract;
import util.management.contractmgr.InterfaceInContract;
import util.management.contractmgr.OpImplementations;
import util.management.interfacemgr.Interface;


public class ContractRegistrator {

	public static void main(String[] args) throws Exception {
		if (args.length != 4) {
			System.err.println("\n Bad arguments. Usage: \n\n " + ContractRegistrator.class.getName()
					+ " <domain_name> <owner_name> <owner_pass> <benef_name> \n");
			return;
		}
		String domainName = args[0];
		String ownerName = args[1];
		String ownerPass = args[2];
		String benefName = args[3];

		AccountID ownerID = ClientManagementLib.getAccountID(ownerName);
		Credential ownerCredential = new PasswordCredential(ownerPass);
		DomainID ownerDomainID = ClientManagementLib.getDomainID(ownerID, ownerCredential, domainName);
		Map<MetaClassID, MetaClass> classesInDomain = ClientManagementLib.getClassesInfoInDomain(ownerID, ownerCredential, ownerDomainID);

		if (classesInDomain != null && classesInDomain.size() > 0) {
			List<InterfaceInContract> newInterfacesInContract = new LinkedList<InterfaceInContract>();

			for (Entry<MetaClassID, MetaClass> curClass : classesInDomain.entrySet()) {
				MetaClassID mclassID = curClass.getKey();
				MetaClass mclass = curClass.getValue();
				System.out.println(" == Current class " + mclass.getName() + " ID " + mclassID);
				Set<String> propertiesNames = new HashSet<String>();
				for (Property prop : mclass.getProperties()) {
					propertiesNames.add(prop.getName());
				}

				Set<String> opsSignature = new HashSet<String>();
				Set<OpImplementations> opImpls = new HashSet<OpImplementations>();
				for (Operation op : mclass.getOperations()) {
					opsSignature.add(op.getSignature());

					OpImplementations newOpImpls = new OpImplementations(op.getSignature(), 0, 0);
					opImpls.add(newOpImpls);
				}

				Interface newIface = new Interface(ownerName, domainName, domainName, mclass.getName(), propertiesNames, opsSignature);

				InterfaceID newIfaceID = ClientManagementLib.newInterface(ownerID, ownerCredential, newIface);
				if (newIfaceID != null) {
					System.out.println(" === Created interface " + newIfaceID);
				} else {
					System.out.println(" === Interface for class " + mclass.getName() + " was not created.");
				}

				InterfaceInContract ifaceInContract = new InterfaceInContract(newIface, opImpls);
				newInterfacesInContract.add(ifaceInContract);
			}

			Set<String> applicants = new HashSet<String>();
			applicants.add(benefName);
			Calendar beginDate = Calendar.getInstance();
			Calendar endDate = Calendar.getInstance();
			beginDate.add(Calendar.YEAR, -1);
			endDate.add(Calendar.YEAR, 1);
			Contract newContract = new Contract(domainName, ownerName, applicants, newInterfacesInContract, beginDate, endDate);
			ContractID newContractID = ClientManagementLib.newPrivateContract(ownerID, ownerCredential, newContract);
			if (newContractID != null) {
				System.out.println(" === Created contract " + newContractID);
			} else {
				System.out.println(" === Contract was not created");
			}
		}

		ClientManagementLib.finishConnections();
	}
}
