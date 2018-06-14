
package es.bsc.genedetection;

import javax.xml.bind.JAXBElement;
import javax.xml.bind.annotation.XmlElementDecl;
import javax.xml.bind.annotation.XmlRegistry;
import javax.xml.namespace.QName;


/**
 * This object contains factory methods for each 
 * Java content interface and Java element interface 
 * generated in the es.bsc.genedetection package. 
 * <p>An ObjectFactory allows you to programatically 
 * construct new instances of the Java representation 
 * for XML content. The Java representation of XML 
 * content can consist of schema derived interfaces 
 * and classes representing the binding of schema 
 * type definitions, element declarations and model 
 * groups.  Factory methods for each of these are 
 * provided in this class.
 * 
 */
@XmlRegistry
public class ObjectFactory {

    private final static QName _DetectGenes_QNAME = new QName("http://genedetection.bsc.es/", "detectGenes");
    private final static QName _DetectGenesResponse_QNAME = new QName("http://genedetection.bsc.es/", "detectGenesResponse");

    /**
     * Create a new ObjectFactory that can be used to create new instances of schema derived classes for package: es.bsc.genedetection
     * 
     */
    public ObjectFactory() {
    }

    /**
     * Create an instance of {@link DetectGenes }
     * 
     */
    public DetectGenes createDetectGenes() {
        return new DetectGenes();
    }

    /**
     * Create an instance of {@link DetectGenesResponse }
     * 
     */
    public DetectGenesResponse createDetectGenesResponse() {
        return new DetectGenesResponse();
    }

    /**
     * Create an instance of {@link JAXBElement }{@code <}{@link DetectGenes }{@code >}}
     * 
     */
    @XmlElementDecl(namespace = "http://genedetection.bsc.es/", name = "detectGenes")
    public JAXBElement<DetectGenes> createDetectGenes(DetectGenes value) {
        return new JAXBElement<DetectGenes>(_DetectGenes_QNAME, DetectGenes.class, null, value);
    }

    /**
     * Create an instance of {@link JAXBElement }{@code <}{@link DetectGenesResponse }{@code >}}
     * 
     */
    @XmlElementDecl(namespace = "http://genedetection.bsc.es/", name = "detectGenesResponse")
    public JAXBElement<DetectGenesResponse> createDetectGenesResponse(DetectGenesResponse value) {
        return new JAXBElement<DetectGenesResponse>(_DetectGenesResponse_QNAME, DetectGenesResponse.class, null, value);
    }

}
