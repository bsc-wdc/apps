
package core.genedetect;

import javax.xml.bind.JAXBElement;
import javax.xml.bind.annotation.XmlElementDecl;
import javax.xml.bind.annotation.XmlRegistry;
import javax.xml.namespace.QName;


/**
 * This object contains factory methods for each 
 * Java content interface and Java element interface 
 * generated in the core.genedetect package. 
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

    private final static QName _FromGenericSequenceToFastaResponse_QNAME = new QName("http://genedetect.core", "fromGenericSequenceToFastaResponse");
    private final static QName _LoadAminoAcidSequence_QNAME = new QName("http://genedetect.core", "loadAminoAcidSequence");
    private final static QName _RunNCBIBlastpResponse_QNAME = new QName("http://genedetect.core", "runNCBIBlastpResponse");
    private final static QName _RunNCBIBlastp_QNAME = new QName("http://genedetect.core", "runNCBIBlastp");
    private final static QName _LoadAminoAcidSequenceResponse_QNAME = new QName("http://genedetect.core", "loadAminoAcidSequenceResponse");
    private final static QName _OverlappingFromBL2G_QNAME = new QName("http://genedetect.core", "overlappingFromBL2G");
    private final static QName _ParseBlastIDsResponse_QNAME = new QName("http://genedetect.core", "parseBlastIDsResponse");
    private final static QName _OverlappingFromBL2GResponse_QNAME = new QName("http://genedetect.core", "overlappingFromBL2GResponse");
    private final static QName _RunBlast2Gene_QNAME = new QName("http://genedetect.core", "runBlast2Gene");
    private final static QName _RunBlast2GeneResponse_QNAME = new QName("http://genedetect.core", "runBlast2GeneResponse");
    private final static QName _ParseBlastIDs_QNAME = new QName("http://genedetect.core", "parseBlastIDs");
    private final static QName _FromGenericSequenceToFasta_QNAME = new QName("http://genedetect.core", "fromGenericSequenceToFasta");

    /**
     * Create a new ObjectFactory that can be used to create new instances of schema derived classes for package: core.genedetect
     * 
     */
    public ObjectFactory() {
    }

    /**
     * Create an instance of {@link LoadAminoAcidSequence }
     * 
     */
    public LoadAminoAcidSequence createLoadAminoAcidSequence() {
        return new LoadAminoAcidSequence();
    }

    /**
     * Create an instance of {@link FromGenericSequenceToFasta }
     * 
     */
    public FromGenericSequenceToFasta createFromGenericSequenceToFasta() {
        return new FromGenericSequenceToFasta();
    }

    /**
     * Create an instance of {@link NemusString }
     * 
     */
    public NemusString createNemusString() {
        return new NemusString();
    }

    /**
     * Create an instance of {@link TextPlain }
     * 
     */
    public TextPlain createTextPlain() {
        return new TextPlain();
    }

    /**
     * Create an instance of {@link AminoAcidSequence }
     * 
     */
    public AminoAcidSequence createAminoAcidSequence() {
        return new AminoAcidSequence();
    }

    /**
     * Create an instance of {@link FromGenericSequenceToFastaResponse }
     * 
     */
    public FromGenericSequenceToFastaResponse createFromGenericSequenceToFastaResponse() {
        return new FromGenericSequenceToFastaResponse();
    }

    /**
     * Create an instance of {@link TextFormatted }
     * 
     */
    public TextFormatted createTextFormatted() {
        return new TextFormatted();
    }

    /**
     * Create an instance of {@link GFF }
     * 
     */
    public GFF createGFF() {
        return new GFF();
    }

    /**
     * Create an instance of {@link ZipEncoded }
     * 
     */
    public ZipEncoded createZipEncoded() {
        return new ZipEncoded();
    }

    /**
     * Create an instance of {@link RunNCBIBlastpSecondaryParameters }
     * 
     */
    public RunNCBIBlastpSecondaryParameters createRunNCBIBlastpSecondaryParameters() {
        return new RunNCBIBlastpSecondaryParameters();
    }

    /**
     * Create an instance of {@link VirtualSequence }
     * 
     */
    public VirtualSequence createVirtualSequence() {
        return new VirtualSequence();
    }

    /**
     * Create an instance of {@link OverlappingFromBL2G }
     * 
     */
    public OverlappingFromBL2G createOverlappingFromBL2G() {
        return new OverlappingFromBL2G();
    }

    /**
     * Create an instance of {@link BL2GAnnotation }
     * 
     */
    public BL2GAnnotation createBL2GAnnotation() {
        return new BL2GAnnotation();
    }

    /**
     * Create an instance of {@link NucleotideSequence }
     * 
     */
    public NucleotideSequence createNucleotideSequence() {
        return new NucleotideSequence();
    }

    /**
     * Create an instance of {@link RunBlast2GeneResponse }
     * 
     */
    public RunBlast2GeneResponse createRunBlast2GeneResponse() {
        return new RunBlast2GeneResponse();
    }

    /**
     * Create an instance of {@link Bl2GAnnotations }
     * 
     */
    public Bl2GAnnotations createBl2GAnnotations() {
        return new Bl2GAnnotations();
    }

    /**
     * Create an instance of {@link BlastIDs }
     * 
     */
    public BlastIDs createBlastIDs() {
        return new BlastIDs();
    }

    /**
     * Create an instance of {@link NemusInteger }
     * 
     */
    public NemusInteger createNemusInteger() {
        return new NemusInteger();
    }

    /**
     * Create an instance of {@link ParseBlastIDs }
     * 
     */
    public ParseBlastIDs createParseBlastIDs() {
        return new ParseBlastIDs();
    }

    /**
     * Create an instance of {@link NemusObject }
     * 
     */
    public NemusObject createNemusObject() {
        return new NemusObject();
    }


    /**
     * Create an instance of {@link BLASTText }
     * 
     */
    public BLASTText createBLASTText() {
        return new BLASTText();
    }

    /**
     * Create an instance of {@link OverlappingFromBL2GResponse }
     * 
     */
    public OverlappingFromBL2GResponse createOverlappingFromBL2GResponse() {
        return new OverlappingFromBL2GResponse();
    }

    /**
     * Create an instance of {@link FASTA }
     * 
     */
    public FASTA createFASTA() {
        return new FASTA();
    }


    /**
     * Create an instance of {@link Annotation }
     * 
     */
    public Annotation createAnnotation() {
        return new Annotation();
    }

    /**
     * Create an instance of {@link NemusFloat }
     * 
     */
    public NemusFloat createNemusFloat() {
        return new NemusFloat();
    }

    /**
     * Create an instance of {@link RunBlast2GeneSecondaryParameters }
     * 
     */
    public RunBlast2GeneSecondaryParameters createRunBlast2GeneSecondaryParameters() {
        return new RunBlast2GeneSecondaryParameters();
    }

    /**
     * Create an instance of {@link NemusReference }
     * 
     */
    public NemusReference createNemusReference() {
        return new NemusReference();
    }

    /**
     * Create an instance of {@link GenericSequence }
     * 
     */
    public GenericSequence createGenericSequence() {
        return new GenericSequence();
    }

    /**
     * Create an instance of {@link LoadAminoAcidSequenceResponse }
     * 
     */
    public LoadAminoAcidSequenceResponse createLoadAminoAcidSequenceResponse() {
        return new LoadAminoAcidSequenceResponse();
    }

    /**
     * Create an instance of {@link ParseBlastIDsResponse }
     * 
     */
    public ParseBlastIDsResponse createParseBlastIDsResponse() {
        return new ParseBlastIDsResponse();
    }

    /**
     * Create an instance of {@link CommentedNASequence }
     * 
     */
    public CommentedNASequence createCommentedNASequence() {
        return new CommentedNASequence();
    }

    /**
     * Create an instance of {@link RunBlast2Gene }
     * 
     */
    public RunBlast2Gene createRunBlast2Gene() {
        return new RunBlast2Gene();
    }

    /**
     * Create an instance of {@link JAXBElement }{@code <}{@link FromGenericSequenceToFastaResponse }{@code >}}
     * 
     */
    @XmlElementDecl(namespace = "http://genedetect.core", name = "fromGenericSequenceToFastaResponse")
    public JAXBElement<FromGenericSequenceToFastaResponse> createFromGenericSequenceToFastaResponse(FromGenericSequenceToFastaResponse value) {
        return new JAXBElement<FromGenericSequenceToFastaResponse>(_FromGenericSequenceToFastaResponse_QNAME, FromGenericSequenceToFastaResponse.class, null, value);
    }

    /**
     * Create an instance of {@link JAXBElement }{@code <}{@link LoadAminoAcidSequence }{@code >}}
     * 
     */
    @XmlElementDecl(namespace = "http://genedetect.core", name = "loadAminoAcidSequence")
    public JAXBElement<LoadAminoAcidSequence> createLoadAminoAcidSequence(LoadAminoAcidSequence value) {
        return new JAXBElement<LoadAminoAcidSequence>(_LoadAminoAcidSequence_QNAME, LoadAminoAcidSequence.class, null, value);
    }

    /**
     * Create an instance of {@link JAXBElement }{@code <}{@link LoadAminoAcidSequenceResponse }{@code >}}
     * 
     */
    @XmlElementDecl(namespace = "http://genedetect.core", name = "loadAminoAcidSequenceResponse")
    public JAXBElement<LoadAminoAcidSequenceResponse> createLoadAminoAcidSequenceResponse(LoadAminoAcidSequenceResponse value) {
        return new JAXBElement<LoadAminoAcidSequenceResponse>(_LoadAminoAcidSequenceResponse_QNAME, LoadAminoAcidSequenceResponse.class, null, value);
    }

    /**
     * Create an instance of {@link JAXBElement }{@code <}{@link OverlappingFromBL2G }{@code >}}
     * 
     */
    @XmlElementDecl(namespace = "http://genedetect.core", name = "overlappingFromBL2G")
    public JAXBElement<OverlappingFromBL2G> createOverlappingFromBL2G(OverlappingFromBL2G value) {
        return new JAXBElement<OverlappingFromBL2G>(_OverlappingFromBL2G_QNAME, OverlappingFromBL2G.class, null, value);
    }

    /**
     * Create an instance of {@link JAXBElement }{@code <}{@link ParseBlastIDsResponse }{@code >}}
     * 
     */
    @XmlElementDecl(namespace = "http://genedetect.core", name = "parseBlastIDsResponse")
    public JAXBElement<ParseBlastIDsResponse> createParseBlastIDsResponse(ParseBlastIDsResponse value) {
        return new JAXBElement<ParseBlastIDsResponse>(_ParseBlastIDsResponse_QNAME, ParseBlastIDsResponse.class, null, value);
    }

    /**
     * Create an instance of {@link JAXBElement }{@code <}{@link OverlappingFromBL2GResponse }{@code >}}
     * 
     */
    @XmlElementDecl(namespace = "http://genedetect.core", name = "overlappingFromBL2GResponse")
    public JAXBElement<OverlappingFromBL2GResponse> createOverlappingFromBL2GResponse(OverlappingFromBL2GResponse value) {
        return new JAXBElement<OverlappingFromBL2GResponse>(_OverlappingFromBL2GResponse_QNAME, OverlappingFromBL2GResponse.class, null, value);
    }

    /**
     * Create an instance of {@link JAXBElement }{@code <}{@link RunBlast2Gene }{@code >}}
     * 
     */
    @XmlElementDecl(namespace = "http://genedetect.core", name = "runBlast2Gene")
    public JAXBElement<RunBlast2Gene> createRunBlast2Gene(RunBlast2Gene value) {
        return new JAXBElement<RunBlast2Gene>(_RunBlast2Gene_QNAME, RunBlast2Gene.class, null, value);
    }

    /**
     * Create an instance of {@link JAXBElement }{@code <}{@link RunBlast2GeneResponse }{@code >}}
     * 
     */
    @XmlElementDecl(namespace = "http://genedetect.core", name = "runBlast2GeneResponse")
    public JAXBElement<RunBlast2GeneResponse> createRunBlast2GeneResponse(RunBlast2GeneResponse value) {
        return new JAXBElement<RunBlast2GeneResponse>(_RunBlast2GeneResponse_QNAME, RunBlast2GeneResponse.class, null, value);
    }

    /**
     * Create an instance of {@link JAXBElement }{@code <}{@link ParseBlastIDs }{@code >}}
     * 
     */
    @XmlElementDecl(namespace = "http://genedetect.core", name = "parseBlastIDs")
    public JAXBElement<ParseBlastIDs> createParseBlastIDs(ParseBlastIDs value) {
        return new JAXBElement<ParseBlastIDs>(_ParseBlastIDs_QNAME, ParseBlastIDs.class, null, value);
    }

    /**
     * Create an instance of {@link JAXBElement }{@code <}{@link FromGenericSequenceToFasta }{@code >}}
     * 
     */
    @XmlElementDecl(namespace = "http://genedetect.core", name = "fromGenericSequenceToFasta")
    public JAXBElement<FromGenericSequenceToFasta> createFromGenericSequenceToFasta(FromGenericSequenceToFasta value) {
        return new JAXBElement<FromGenericSequenceToFasta>(_FromGenericSequenceToFasta_QNAME, FromGenericSequenceToFasta.class, null, value);
    }

}
