package randomforest.client;

import javax.xml.bind.JAXBContext;
import javax.xml.bind.JAXBException;
import javax.xml.bind.Marshaller;
import javax.xml.bind.PropertyException;


public class Debugger {

    private static final boolean DEBUG = true;

    public static void out(String tag, String message) {
        System.out.println("[" + tag.toUpperCase() + "] " + message);
    }

    public static void out(String message) {
        System.out.println(message);
    }

    public static final void out(Exception e) {
        e.printStackTrace(System.out);
    }

    public static final void debug(Exception e) {
        if (DEBUG) {
            e.printStackTrace(System.out);
        }
    }

    public static final void debug(String tag, String message) {
        if (DEBUG) {
            System.out.println("[" + tag.toUpperCase() + "] " + message);
        }
    }

    public static final void debug(String message) {
        if (DEBUG) {
            System.out.println(message);
        }
    }

    public static final void err(String tag, String message) {
        System.err.println("[" + tag.toUpperCase() + "] " + message);
    }

    public static final void err(String message) {
        System.err.println(message);
    }

    public static final void err(Exception e) {
        e.printStackTrace(System.err);
    }

    public static void debugAsXML(Object value) throws PropertyException, JAXBException {
        if (DEBUG) {
            JAXBContext jaxbContext = JAXBContext.newInstance(value.getClass());
            Marshaller jaxbMarshaller = jaxbContext.createMarshaller();
            jaxbMarshaller.setProperty(Marshaller.JAXB_FORMATTED_OUTPUT, true);
            jaxbMarshaller.marshal(value, System.out);
        }
    }
}
