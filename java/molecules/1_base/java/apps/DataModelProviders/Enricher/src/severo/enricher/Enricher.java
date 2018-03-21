package severo.enricher;

import java.util.*;
import java.util.Map.Entry;

import specs.*;
import specs.contractmgr.*;
import util.ids.*;
import util.credentials.*;
import util.language.SupportedLanguages.Langs;
import info.classmgr.*;
import info.contractmgr.ContractInfo;
import info.interfacemgr.InterfaceInfo;
import util.structs.Tuple;
import client.ClientManagementLib;

public class Enricher {

    private static DomainID baseModelDomainID;

    private static final String domainName = "EnricherDomain";
    private static DomainID enricherDomainID;

	private static final String originalClassName = "severo.molecule.Molecule";
	private static final String className = "severo.molecule.EnrichedMolecule";
	private static final String classPath = "./Stubs/Enricher/severo/molecule";

	private static DomainID domainIDofProvider;
	private static EnrichmentInfo enrichmentInfo;
	private static Hashtable<InterfaceID, InterfaceInContract> interfacesInContract;


	public static void main(String[] args) throws Exception {
        // Get ID of the domain that provides base model
		baseModelDomainID = ClientManagementLib.getDomainID(AccountInfo.ENRICHER.accountID, AccountInfo.ENRICHER.password, "RegistratorDomain");
	
        // Register the enrichments of the base model	
        if (!registerEnrichment()) {
			System.err.println("Classes could not be created or already registered.");
            System.exit(-1);
        }

        // Register interfaces for the model enriched
		if (!registerInterfaceForContracts()) {
            System.err.println("Interface for contracts could not be created.");
            System.exit(-1);
        }
		
        // Register contract of Enricher with himself to make new methods usable
		if (registerContract(AccountInfo.ENRICHER) == null) {
            System.err.println("Contract for Producer cannot be created.");
            System.exit(-1);
        }
			
		// Register contract between Enricher and Consumer to enable Consumer to execute enrichments
		if (registerContract(AccountInfo.CONSUMER) == null) {
            System.err.println("Contract for Consumer cannot be created.");
            System.exit(-1);
        }
	}



	private static boolean registerEnrichment() throws Exception {
		// Create a new Domain
		System.out.println("Creating the domain of the enricher...");
		enricherDomainID = ClientManagementLib.newDomain(AccountInfo.ENRICHER.accountID, AccountInfo.ENRICHER.password, domainName, Langs.LANG_JAVA);
		if (enricherDomainID == null) {
			enricherDomainID = ClientManagementLib.getDomainID(AccountInfo.ENRICHER.accountID, AccountInfo.ENRICHER.password, domainName);
			if (enricherDomainID == null) {
				return false;
			}
		}

		// Import enricher contract to the new Domain
		Hashtable<ContractID, ContractInfo> enricherContracts = ClientManagementLib.getContractsOfApplicant(
            AccountInfo.ENRICHER.accountID, AccountInfo.ENRICHER.password, baseModelDomainID);

		ContractID enricherContractID = enricherContracts.keySet().iterator().next();

		domainIDofProvider = enricherContracts.get(enricherContractID).getProviderDomainID();

		ClientManagementLib.importContract(AccountInfo.ENRICHER.accountID, AccountInfo.ENRICHER.password, enricherDomainID, enricherContractID);
		System.out.println("Imported Contract between RegistratorDomain and Enricher into EnricherDomain");

		// Create Enrichment
		Tuple<EnrichmentInfo, Hashtable<String, MetaClassInfo>> 
			enrichmentAllInfo = ClientManagementLib.newEnrichment(
				AccountInfo.ENRICHER.accountID, AccountInfo.ENRICHER.password, 
				enricherDomainID, originalClassName, className, classPath);
		
		System.out.println("Created the enrichment for class " + originalClassName + " by using " + className);

		enrichmentInfo = enrichmentAllInfo.getFirst();

		return true;
	}


	
	private static boolean registerInterfaceForContracts() throws Exception {
        interfacesInContract = new Hashtable<InterfaceID, InterfaceInContract>();

		// Prepare properties in interface
		HashSet<PropertyID> propertiesIDsInInterface = new HashSet<PropertyID>();
		for (PropertyInfo propertyInfo : enrichmentInfo.getProperties().values()) {
			System.out.println("Enriched property: " + propertyInfo.getName());
			propertiesIDsInInterface.add(propertyInfo.getID());
		}

		// Prepare operations and implementations in interface
		HashSet<OperationID> operationsIDsInInterface = new HashSet<OperationID>();
		HashMap<OperationID, OpImplementations> operationImplementations = new HashMap<OperationID, OpImplementations>();
		for (OperationInfo operationInfo : enrichmentInfo.getOperations().values()) {
			System.out.println("Enriched operation: " + operationInfo.getSignature());
			operationsIDsInInterface.add(operationInfo.getID());
			HashSet<ImplementationID> implementationIDsOfOperation = 
				new HashSet<ImplementationID>(operationInfo.getImplementations().keySet());
			ImplementationID implIDInInterface = implementationIDsOfOperation.iterator().next(); //get the first implementation ID
			operationImplementations.put(operationInfo.getID(), 
				new OpImplementations(implIDInInterface, implIDInInterface)); //local and remote implementations	
		}

		// Creates the interface 
	    System.out.println("Creating interface for the enriched class");
		InterfaceID interfaceID = ClientManagementLib.newInterface(
            AccountInfo.ENRICHER.accountID, AccountInfo.ENRICHER.password, 
            enricherDomainID, originalClassName, operationsIDsInInterface, propertiesIDsInInterface);
        if (interfaceID == null) {
            return false;
        }
			
		// Now create the interface in contract
	    System.out.println("Creating interface in contract");
	    
		MetaClassID classIDofInterface = ClientManagementLib.getClassID(AccountInfo.REGISTRATOR.accountID, AccountInfo.REGISTRATOR.password, domainIDofProvider, originalClassName); 
		InterfaceInContract theInterfaceInContract = new InterfaceInContract(interfaceID, classIDofInterface, operationImplementations);
		interfacesInContract.put(interfaceID, theInterfaceInContract);
        
        return interfacesInContract.size() > 0;
	}



	private static ContractID registerContract(AccountInfo user) throws Exception {
		System.out.println("Registering contract for user : " + user.userName);
		
        // Prepare dates of the contract 
		Calendar beginDateOfContract = Calendar.getInstance(); //now
		Calendar endDateOfContract = Calendar.getInstance(); //next year
		endDateOfContract.add(Calendar.YEAR, 1);

		// Create contract spec
		ContractSpec contractSpec = new ContractSpec(interfacesInContract, beginDateOfContract, endDateOfContract);

		// Register the contract 
		return ClientManagementLib.newPrivateContract(AccountInfo.ENRICHER.accountID, AccountInfo.ENRICHER.password,
				user.accountID, enricherDomainID, contractSpec);
	}


    /** Info of accounts */
    private enum AccountInfo {
		REGISTRATOR("Registrator", "Registrator"),
        ENRICHER("Enricher", "Enricher"),
        CONSUMER("Consumer", "Consumer");

		AccountInfo(String theUserName, String thePassword) {
			try {
				userName = theUserName;
				password = new PasswordCredential(thePassword);
				accountID = ClientManagementLib.getAccountID(theUserName);
				if (accountID == null) {
					System.err.println("Bad account : " + theUserName);
					System.exit(-1);
				}
			} catch (Exception ex) {
				System.err.println("Could not get account info");
				System.exit(-1);
			}
        }
        
        public String userName;
        public AccountID accountID;
        public PasswordCredential password;
        
    }
}
