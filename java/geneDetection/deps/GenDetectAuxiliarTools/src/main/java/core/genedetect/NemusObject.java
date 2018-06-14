
package core.genedetect;

import java.util.ArrayList;
import java.util.List;
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlSeeAlso;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for NemusObject complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="NemusObject">
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;sequence>
 *         &lt;element name="reference" type="{http://genedetect.core}NemusReference" maxOccurs="unbounded" minOccurs="0"/>
 *       &lt;/sequence>
 *       &lt;attribute name="nemusId" type="{http://www.w3.org/2001/XMLSchema}string" />
 *       &lt;attribute name="nemusNamespace" type="{http://www.w3.org/2001/XMLSchema}string" />
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "NemusObject", propOrder = {
    "reference"
})
@XmlSeeAlso({
    TextPlain.class,
    ZipEncoded.class,
    Annotation.class,
    VirtualSequence.class
})
public class NemusObject {

    @XmlElement(nillable = true)
    protected List<NemusReference> reference;
    @XmlAttribute(name = "nemusId")
    protected String nemusId;
    @XmlAttribute(name = "nemusNamespace")
    protected String nemusNamespace;

    /**
     * Gets the value of the reference property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the reference property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getReference().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link NemusReference }
     * 
     * 
     */
    public List<NemusReference> getReference() {
        if (reference == null) {
            reference = new ArrayList<NemusReference>();
        }
        return this.reference;
    }

    /**
     * Gets the value of the nemusId property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getNemusId() {
        return nemusId;
    }

    /**
     * Sets the value of the nemusId property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setNemusId(String value) {
        this.nemusId = value;
    }

    /**
     * Gets the value of the nemusNamespace property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getNemusNamespace() {
        return nemusNamespace;
    }

    /**
     * Sets the value of the nemusNamespace property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setNemusNamespace(String value) {
        this.nemusNamespace = value;
    }

}
