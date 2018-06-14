package license.test;

import java.io.File;
import java.io.FileInputStream;
import java.net.URL;
import java.security.cert.CertificateFactory;
import java.security.cert.X509Certificate;

import eu.elasticlm.api.LicenseEnforcement;
import eu.elasticlm.api.utils.TokenFactory;
import eu.elasticlm.schemas.x2009.x05.license.token.FeatureType;
import eu.elasticlm.schemas.x2009.x05.license.token.LicenseTokenDocument;
import eu.elasticlm.schemas.x2009.x05.license.token.LicenseTokenType;
import eu.elasticlm.schemas.x2009.x05.lsdl.ApplicationType;
import eu.elasticlm.schemas.x2009.x05.lsdl.ChargeType;
import eu.elasticlm.schemas.x2009.x05.lsdl.ConsumableFeatureType;
import eu.elasticlm.schemas.x2009.x05.lsdl.CurrencyType;
import eu.elasticlm.schemas.x2009.x05.lsdl.FeaturesType;
import eu.elasticlm.schemas.x2009.x05.lsdl.LicenseDescriptionDocument;
import eu.elasticlm.schemas.x2009.x05.lsdl.LicenseDescriptionType;

public class TokenCheckTest {
	private static final String THREADS_FEATURE_NAME = "THREADS";
	private static final String VERSION = "1.0";
	private static final String APPNAME_FEATURE_ID = "app-name";
	private static final String THREADS_FEATURE_ID = "cfthreads";
	private static final String THREADS_FEATURE = "THREADS";
	public static boolean checkLicenseToken() {
        try {
        	CertificateFactory cf = CertificateFactory.getInstance("X.509");
        	X509Certificate isvCertificate = (X509Certificate) cf.generateCertificate(TokenCheckTest.class.getResourceAsStream("Demo_ISV.pem"));
        	URL tokenURL = new File("/mnt/context/licensetoken/license.token.0").toURI().toURL();
            LicenseEnforcement tokenValidator = LicenseEnforcement.Factory.newInstance(tokenURL, isvCertificate, "/optimis_service/");
            if (tokenValidator.hasFeature(THREADS_FEATURE_NAME)){
                    System.out.println(tokenValidator.getFeature(THREADS_FEATURE_NAME).getValue());
            		if (Integer.parseInt(tokenValidator.getFeature(THREADS_FEATURE_NAME).getValue()) > 6)
            			return true;
            		else
            			return false;
            }else
            	return false;
            
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
	}
	public static String generateToken(String licenseName, String lsERP, String cliProp, int numThreads) throws Exception {
		LicenseTokenDocument licenseTokenDoc = null;
		LicenseDescriptionDocument lsdlDoc = LicenseDescriptionDocument.Factory.newInstance();
		LicenseDescriptionType lsdl = lsdlDoc.addNewLicenseDescription();
		lsdl.setIsUsedOffline(false);
		ApplicationType appType = lsdl.addNewApplication();
		appType.setApplicationId(licenseName);
		appType.setName(licenseName);
		appType.setVersion(VERSION);
		FeaturesType features = appType.addNewFeatures();
		ConsumableFeatureType basicFeature = features.addNewBasicFeature();
		basicFeature.setFeatureId(APPNAME_FEATURE_ID);
		basicFeature.setName(licenseName);
		basicFeature.setVersion(VERSION);
		basicFeature.setValue(1);
		ConsumableFeatureType feature = features.addNewConsumableFeature();
		feature.setFeatureId(THREADS_FEATURE_ID);
		feature.setName(THREADS_FEATURE);
		feature.setVersion(VERSION);
		feature.setValue(numThreads);
		/*ReservationTimeType res = lsdl.addNewReservation();
		res.setDuration(new GDuration("PT2H"));
		Calendar now = Calendar.getInstance();
		
		now.setTime(new Date("2012-10-01T15:00:00"));
		Calendar deadline = Calendar.getInstance();
		deadline.setTime(new Date("2015-12-31T00:00:00"));
		res.setEarliestStartTime(now);
		res.setDeadline(deadline);
		res.setReservationStartTime(now);
		now.add(Calendar.HOUR_OF_DAY, 5);
		res.setReservationEndTime(now);*/
		ChargeType charge = lsdl.addNewCharge();
		charge.setCurrency(CurrencyType.EUR);
		charge.setStringValue("0");
		lsdl.setAccountingGroup("/ElasticLM/Users");
		
		licenseTokenDoc = TokenFactory.createToken(lsdl, lsERP, cliProp);
		return licenseTokenDoc.toString();
     
	}
	
	public static String getName(String strToken) throws Exception{
		LicenseTokenDocument doc = LicenseTokenDocument.Factory.parse(strToken);
		LicenseTokenType token = doc.getLicenseToken();
		if 	(token!=null){
			for (FeatureType ft:token.getFeatures().getFeatureArray()){
				if (ft.getFeatureId().equals(APPNAME_FEATURE_ID)){
					return ft.getName();
				}
			}
			throw new Exception("Application name feature not found");
		}else
			throw new Exception("Unable to parse license token");
	}
	
	public static void main(String[] args) {
        try {
        	
        	generateToken("Blast", "http://optimis-lms.ds.cs.umu.se:8080/elasticlm-license-service", "/home/jorgee/Projects/Optimis/LicenseTokens/client.properties", 10 );
            
        } catch (Exception e) {
            e.printStackTrace();
        }
	}
	/*public static void main(String[] args) {
        try {
        	CertificateFactory cf = CertificateFactory.getInstance("X.509");
        	X509Certificate isvCertificate = (X509Certificate) cf.generateCertificate(TokenCheckTest.class.getResourceAsStream("Demo_ISV.pem"));
        	URL tokenURL = new File("/home/jorgee/Projects/Optimis/LicenseTokens/blast-token.xml").toURI().toURL();
            LicenseEnforcement tokenValidator = LicenseEnforcement.Factory.newInstance(tokenURL, isvCertificate, "/home/jorgee/Projects/Optimis/LicenseTokens/");
            if (tokenValidator.hasFeature(THREADS_FEATURE_NAME))
                    System.out.println(tokenValidator.getFeature(THREADS_FEATURE_NAME).getValue());
            
            
        } catch (Exception e) {
            e.printStackTrace();
        }
	}*/

}
