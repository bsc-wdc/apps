package severo.registrator;

import java.util.Calendar;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Hashtable;
import java.util.Map;
import java.util.Map.Entry;

import util.credentials.PasswordCredential;
import util.ids.AccountID;
import util.ids.ContractID;
import util.ids.DomainID;
import util.ids.ImplementationID;
import util.ids.InterfaceID;
import util.ids.MetaClassID;
import util.ids.OperationID;
import util.ids.PropertyID;
import util.language.SupportedLanguages.Langs;
import util.management.classmgr.Implementation;
import util.management.classmgr.MetaClass;
import util.management.classmgr.Operation;
import util.management.classmgr.Property;
import util.management.contractmgr.Contract;
import util.management.contractmgr.InterfaceInContract;
import util.management.contractmgr.OpImplementations;
import client.ClientManagementLib;

public class Registrator {
	private static final String domainName = "RegistratorDomain";
	private static DomainID domainID;

	private static Map<String, MetaClass> registeredClasses;
	private static Hashtable<InterfaceID, InterfaceInContract> interfacesInContract;
	//private static final String[] classNames = { "severo.molecule.Molecule" };
	//private static final String classPath = "./DataModels/Molecule/lib/molecule.jar";
	private static String[] classNames = null;
	private static String classPath = null;
	
	public static void main(final String[] args) throws Exception {

		classNames = new String[]{args[0]};
		classPath = args[1];
		
		// Register domain
		if (!registerDomain()) {
			System.err.println("Domain could not be created.");
			ClientManagementLib.finishConnections();
			System.exit(-1);
		}

		registeredClasses = new Hashtable<String, MetaClass>();

		// Register base data model
		for (String className : classNames) {
			if (!registerClass(className)) {
				System.err
						.println("Classes could not be created or already registered.");
				ClientManagementLib.finishConnections();
				System.exit(-1);
			}
		}

		// Register interfaces of the base data model
		if (!registerInterfacesForContracts()) {
			System.err
					.println("Interfaces for contracts could not be created.");
			ClientManagementLib.finishConnections();
			System.exit(-1);
		}

		// Registers contract with Registrator itself in order to make
		// implementations to be executable
		if (registerContract(AccountInfo.REGISTRATOR) == null) {
			System.err.println("Contract with Registrator cannot be created.");
			ClientManagementLib.finishConnections();
			System.exit(-1);
		}

		// Registers contract with Enricher (who will enrich it)
		if (registerContract(AccountInfo.ENRICHER) == null) {
			System.err.println("Contract with Enricher cannot be created.");
			ClientManagementLib.finishConnections();
			System.exit(-1);
		}

		// Registers contract with Producer (who will use it)
		if (registerContract(AccountInfo.PRODUCER) == null) {
			System.err.println("Contract with Producer cannot be created.");
			ClientManagementLib.finishConnections();
			System.exit(-1);
		}

		// Registers contract with Consumer (who will use it)
		if (registerContract(AccountInfo.CONSUMER) == null) {
			System.err.println("Contract with Consumer cannot be created.");
			ClientManagementLib.finishConnections();
			System.exit(-1);
		}

		ClientManagementLib.finishConnections();

	}

	private static boolean registerDomain() throws Exception {
		// Create a new Domain
		domainID = ClientManagementLib.newDomain(
				AccountInfo.REGISTRATOR.accountID,
				AccountInfo.REGISTRATOR.password, domainName, Langs.LANG_JAVA);
		if (domainID == null) {
			domainID = ClientManagementLib.getDomainID(
					AccountInfo.REGISTRATOR.accountID,
					AccountInfo.REGISTRATOR.password, domainName);
			if (domainID == null) {
				System.out.println("Domain " + domainName
						+ " could not be registered.");
				return false;
			}
		}
		return true;
	}

	private static boolean registerClass(final String className)
			throws Exception {
		// Add the new class
		Map<String, MetaClass> result = ClientManagementLib.newClass(
				AccountInfo.REGISTRATOR.accountID,
				AccountInfo.REGISTRATOR.password, domainID, className,
				classPath);

		registeredClasses.putAll(result);

		return registeredClasses.size() > 0;
	}

	private static boolean registerInterfacesForContracts() {
		interfacesInContract = new Hashtable<InterfaceID, InterfaceInContract>();

		// Register an interface for each associated class.
		for (Entry<String, MetaClass> registeredClassEntry : registeredClasses
				.entrySet()) {
			System.out.println("Creating an interface for class: "
					+ registeredClassEntry.getKey());

			String className = registeredClassEntry.getKey();
			MetaClass registeredClass = registeredClassEntry.getValue();

			// Prepare properties in interface
			HashSet<PropertyID> propertiesIDsInInterface = new HashSet<PropertyID>();
			for (Property propertyInfo : registeredClass.getProperties()) {
				propertiesIDsInInterface.add(propertyInfo.getDataClayID());
			}

			// Prepare operations and implementations in interface
			HashSet<OperationID> operationsIDsInInterface = new HashSet<OperationID>();
			HashMap<OperationID, OpImplementations> operationImplementations = new HashMap<OperationID, OpImplementations>();
			for (Operation operationInfo : registeredClass.getOperations()) {
				operationsIDsInInterface.add(operationInfo.getDataClayID());
				HashSet<ImplementationID> implementationIDsOfOperation = new HashSet<ImplementationID>();
				for (Implementation impl : operationInfo.getImplementations()) {
					implementationIDsOfOperation.add(impl.getDataClayID());
				}
				ImplementationID implIDInInterface = implementationIDsOfOperation
						.iterator().next(); // get the first implementation ID
				operationImplementations.put(operationInfo.getID(),
						new OpImplementations(implIDInInterface,
								implIDInInterface)); // local and remote
														// implementations
			}

			// Registers the interface
			InterfaceID interfaceID = ClientManagementLib.newInterface(
					AccountInfo.REGISTRATOR.accountID,
					AccountInfo.REGISTRATOR.password, domainID, className,
					operationsIDsInInterface, propertiesIDsInInterface);

			// Now create the interface in contract
			MetaClassID classIDofInterface = ClientManagementLib.getClassID(
					AccountInfo.REGISTRATOR.accountID,
					AccountInfo.REGISTRATOR.password, domainID, className);
			InterfaceInContract theInterfaceInContract = new InterfaceInContract(
					interfaceID, classIDofInterface, operationImplementations);
			interfacesInContract.put(interfaceID, theInterfaceInContract);
		}

		return interfacesInContract.size() > 0;
	}

	private static ContractID registerContract(final AccountInfo user) {
		System.out.println("Registering contract for user : " + user.userName);

		// Prepare dates of the contract
		Calendar beginDateOfContract = Calendar.getInstance(); // now
		Calendar endDateOfContract = Calendar.getInstance(); // next year
		endDateOfContract.add(Calendar.YEAR, 1);

		// Create contract spec
		Contract contract = new Contract(domainID, user.accountID,
				interfacesInContract, beginDateOfContract, endDateOfContract);

		// Register the contract
		return ClientManagementLib.newPrivateContract(
				AccountInfo.REGISTRATOR.accountID,
				AccountInfo.REGISTRATOR.password, contract);
	}

	/** Info of accounts */
	private enum AccountInfo {
		REGISTRATOR("Registrator", "Registrator"), ENRICHER("Enricher",
				"Enricher"), PRODUCER("Producer", "Producer"), CONSUMER(
				"Consumer", "Consumer");

		AccountInfo(final String theUserName, final String thePassword) {
			try {
				this.userName = theUserName;
				this.password = new PasswordCredential(thePassword);
				this.accountID = ClientManagementLib.getAccountID(theUserName);
				if (accountID == null) {
					System.err.println("Bad account : " + theUserName);
					System.exit(-1);
				}
			} catch (Exception ex) {
				System.err.println("Could not initialize accounts' info");
				System.exit(-1);
			}
		}

		public String userName;
		public AccountID accountID;
		public PasswordCredential password;

	}
}
