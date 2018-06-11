
package core.genedetect;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for RunBlast2GeneSecondaryParameters complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="RunBlast2GeneSecondaryParameters">
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;all>
 *         &lt;element name="coverage" type="{http://www.w3.org/2001/XMLSchema}float" minOccurs="0"/>
 *         &lt;element name="geneonly" type="{http://www.w3.org/2001/XMLSchema}boolean" minOccurs="0"/>
 *       &lt;/all>
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "RunBlast2GeneSecondaryParameters", propOrder = {

})
public class RunBlast2GeneSecondaryParameters {

    protected Float coverage;
    protected Boolean geneonly;

    /**
     * Gets the value of the coverage property.
     * 
     * @return
     *     possible object is
     *     {@link Float }
     *     
     */
    public Float getCoverage() {
        return coverage;
    }

    /**
     * Sets the value of the coverage property.
     * 
     * @param value
     *     allowed object is
     *     {@link Float }
     *     
     */
    public void setCoverage(Float value) {
        this.coverage = value;
    }

    /**
     * Gets the value of the geneonly property.
     * 
     * @return
     *     possible object is
     *     {@link Boolean }
     *     
     */
    public Boolean isGeneonly() {
        return geneonly;
    }

    /**
     * Sets the value of the geneonly property.
     * 
     * @param value
     *     allowed object is
     *     {@link Boolean }
     *     
     */
    public void setGeneonly(Boolean value) {
        this.geneonly = value;
    }

}
