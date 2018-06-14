
package core.genedetect;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlSeeAlso;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for VirtualSequence complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="VirtualSequence">
 *   &lt;complexContent>
 *     &lt;extension base="{http://genedetect.core}NemusObject">
 *       &lt;sequence>
 *         &lt;element name="Length" type="{http://genedetect.core}NemusInteger" minOccurs="0"/>
 *       &lt;/sequence>
 *     &lt;/extension>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "VirtualSequence", propOrder = {
    "length"
})
@XmlSeeAlso({
    GenericSequence.class
})
public class VirtualSequence
    extends NemusObject
{

    @XmlElement(name = "Length")
    protected NemusInteger length;

    /**
     * Gets the value of the length property.
     * 
     * @return
     *     possible object is
     *     {@link NemusInteger }
     *     
     */
    public NemusInteger getLength() {
        return length;
    }

    /**
     * Sets the value of the length property.
     * 
     * @param value
     *     allowed object is
     *     {@link NemusInteger }
     *     
     */
    public void setLength(NemusInteger value) {
        this.length = value;
    }

}
