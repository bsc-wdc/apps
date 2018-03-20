package tools;

import java.util.Map;

import client.ClientManagementLib;
import util.ids.AccountID;
import util.ids.DomainID;
import util.language.SupportedLanguages.Langs;
import util.management.accountmgr.Credential;
import util.management.accountmgr.PasswordCredential;
import util.management.classmgr.MetaClass;
import util.management.domainmgr.Domain;


public class ClassRegistrator {

	public static void main(String[] args) throws Exception {
		if (args.length != 5) {
			System.err.println("\n Bad arguments. Usage: \n\n " + ClassRegistrator.class.getName()
					+ " <domain_name> <class_name> <class_path> <owner_name> <owner_pass> \n");
			return;
		}
		String domainName = args[0];
		String className = args[1];
		String classPath = args[2];
		String ownerName = args[3];
		String ownerPass = args[4];

		// Get accounts info
		System.out.println(" == Getting account info for : " + ownerName);
		AccountID ownerID = ClientManagementLib.getAccountID(ownerName);
		Credential ownerCredential = new PasswordCredential(ownerPass);

		// Register namespace
		System.out.println(" == Checking namespace " + domainName);
		DomainID ownerDomainID = ClientManagementLib.getDomainID(ownerID, ownerCredential, domainName);
		if (ownerDomainID == null) {
			System.out.println(" ==== Namespace not found, creating it ...");
			Domain d = new Domain(domainName, ownerName, Langs.LANG_JAVA);
			ownerDomainID = ClientManagementLib.newDomain(ownerID, ownerCredential, d);
		}
		System.out.println(" == Namespace id: " + ownerDomainID);

		// Register the included classes
		Map<String, MetaClass> registeredClasses = null;

		MetaClass mclass = ClientManagementLib.getClassInfo(ownerID, ownerCredential, ownerDomainID, className);
		if (mclass == null) {
			System.out.println(" == Registering class " + className + " using classpath: " + classPath);
			registeredClasses = ClientManagementLib.newClass(ownerID, ownerCredential, domainName, className, classPath);
			System.out.println(" == Registered classes: \n " + registeredClasses.keySet());
		} else {
			System.out.println(" == Class already registered " + mclass.getDataClayID());
		}
	}
}
