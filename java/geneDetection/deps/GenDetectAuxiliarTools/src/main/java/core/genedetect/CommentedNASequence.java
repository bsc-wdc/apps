
package core.genedetect;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for CommentedNASequence complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="CommentedNASequence">
 *   &lt;complexContent>
 *     &lt;extension base="{http://genedetect.core}NucleotideSequence">
 *       &lt;sequence>
 *         &lt;element name="Description" type="{http://genedetect.core}NemusString" minOccurs="0"/>
 *       &lt;/sequence>
 *     &lt;/extension>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "CommentedNASequence", propOrder = {
    "description"
})
public class CommentedNASequence
    extends NucleotideSequence
{

    @XmlElement(name = "Description")
    protected NemusString description;

    /**
     * Gets the value of the description property.
     * 
     * @return
     *     possible object is
     *     {@link NemusString }
     *     
     */
    public NemusString getDescription() {
        return description;
    }

    /**
     * Sets the value of the description property.
     * 
     * @param value
     *     allowed object is
     *     {@link NemusString }
     *     
     */
    public void setDescription(NemusString value) {
        this.description = value;
    }

}
