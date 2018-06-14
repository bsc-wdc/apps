
package core.genedetect;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for parseBlastIDs complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="parseBlastIDs">
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;sequence>
 *         &lt;element name="report" type="{http://genedetect.core}BLAST-Text" minOccurs="0"/>
 *       &lt;/sequence>
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "parseBlastIDs", propOrder = {
    "report"
})
public class ParseBlastIDs {

    protected BLASTText report;

    /**
     * Gets the value of the report property.
     * 
     * @return
     *     possible object is
     *     {@link BLASTText }
     *     
     */
    public BLASTText getReport() {
        return report;
    }

    /**
     * Sets the value of the report property.
     * 
     * @param value
     *     allowed object is
     *     {@link BLASTText }
     *     
     */
    public void setReport(BLASTText value) {
        this.report = value;
    }

}
