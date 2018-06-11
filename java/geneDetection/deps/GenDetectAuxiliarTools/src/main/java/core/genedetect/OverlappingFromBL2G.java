
package core.genedetect;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for overlappingFromBL2G complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="overlappingFromBL2G">
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;sequence>
 *         &lt;element name="ovAnnots" type="{http://genedetect.core}bl2GAnnotations" minOccurs="0"/>
 *       &lt;/sequence>
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "overlappingFromBL2G", propOrder = {
    "ovAnnots"
})
public class OverlappingFromBL2G {

    protected Bl2GAnnotations ovAnnots;

    /**
     * Gets the value of the ovAnnots property.
     * 
     * @return
     *     possible object is
     *     {@link Bl2GAnnotations }
     *     
     */
    public Bl2GAnnotations getOvAnnots() {
        return ovAnnots;
    }

    /**
     * Sets the value of the ovAnnots property.
     * 
     * @param value
     *     allowed object is
     *     {@link Bl2GAnnotations }
     *     
     */
    public void setOvAnnots(Bl2GAnnotations value) {
        this.ovAnnots = value;
    }

}
