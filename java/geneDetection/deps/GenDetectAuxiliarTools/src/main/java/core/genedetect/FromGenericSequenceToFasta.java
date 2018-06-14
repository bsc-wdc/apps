
package core.genedetect;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for fromGenericSequenceToFasta complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="fromGenericSequenceToFasta">
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;sequence>
 *         &lt;element name="seqId" type="{http://genedetect.core}NemusObject" minOccurs="0"/>
 *       &lt;/sequence>
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "fromGenericSequenceToFasta", propOrder = {
    "seqId"
})
public class FromGenericSequenceToFasta {

    protected NemusObject seqId;

    /**
     * Gets the value of the seqId property.
     * 
     * @return
     *     possible object is
     *     {@link NemusObject }
     *     
     */
    public NemusObject getSeqId() {
        return seqId;
    }

    /**
     * Sets the value of the seqId property.
     * 
     * @param value
     *     allowed object is
     *     {@link NemusObject }
     *     
     */
    public void setSeqId(NemusObject value) {
        this.seqId = value;
    }

}
