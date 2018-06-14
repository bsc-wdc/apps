
package core.genedetect;

import java.util.ArrayList;
import java.util.List;
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for bl2GAnnotations complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="bl2GAnnotations">
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;sequence>
 *         &lt;element name="annots" type="{http://genedetect.core}BL2GAnnotation" maxOccurs="unbounded" minOccurs="0"/>
 *       &lt;/sequence>
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "bl2GAnnotations", propOrder = {
    "annots"
})
public class Bl2GAnnotations {

    @XmlElement(nillable = true)
    protected List<BL2GAnnotation> annots;

    /**
     * Gets the value of the annots property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the annots property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getAnnots().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link BL2GAnnotation }
     * 
     * 
     */
    public List<BL2GAnnotation> getAnnots() {
        if (annots == null) {
            annots = new ArrayList<BL2GAnnotation>();
        }
        return this.annots;
    }

}
