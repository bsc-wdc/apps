
package core.genedetect;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlSeeAlso;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for GenericSequence complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="GenericSequence">
 *   &lt;complexContent>
 *     &lt;extension base="{http://genedetect.core}VirtualSequence">
 *       &lt;sequence>
 *         &lt;element name="SequenceString" type="{http://genedetect.core}NemusString" minOccurs="0"/>
 *       &lt;/sequence>
 *     &lt;/extension>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "GenericSequence", propOrder = {
    "sequenceString"
})
@XmlSeeAlso({
    AminoAcidSequence.class,
    NucleotideSequence.class
})
public class GenericSequence
    extends VirtualSequence
{

    @XmlElement(name = "SequenceString")
    protected NemusString sequenceString;

    /**
     * Gets the value of the sequenceString property.
     * 
     * @return
     *     possible object is
     *     {@link NemusString }
     *     
     */
    public NemusString getSequenceString() {
        return sequenceString;
    }

    /**
     * Sets the value of the sequenceString property.
     * 
     * @param value
     *     allowed object is
     *     {@link NemusString }
     *     
     */
    public void setSequenceString(NemusString value) {
        this.sequenceString = value;
    }

}
