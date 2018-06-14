
package core.genedetect;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for Zip_Encoded complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="Zip_Encoded">
 *   &lt;complexContent>
 *     &lt;extension base="{http://genedetect.core}NemusObject">
 *       &lt;sequence>
 *         &lt;element name="mimeTypte" type="{http://genedetect.core}NemusString" minOccurs="0"/>
 *         &lt;element name="rawdata" type="{http://genedetect.core}NemusString" minOccurs="0"/>
 *       &lt;/sequence>
 *     &lt;/extension>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "Zip_Encoded", propOrder = {
    "mimeTypte",
    "rawdata"
})
public class ZipEncoded
    extends NemusObject
{

    protected NemusString mimeTypte;
    protected NemusString rawdata;

    /**
     * Gets the value of the mimeTypte property.
     * 
     * @return
     *     possible object is
     *     {@link NemusString }
     *     
     */
    public NemusString getMimeTypte() {
        return mimeTypte;
    }

    /**
     * Sets the value of the mimeTypte property.
     * 
     * @param value
     *     allowed object is
     *     {@link NemusString }
     *     
     */
    public void setMimeTypte(NemusString value) {
        this.mimeTypte = value;
    }

    /**
     * Gets the value of the rawdata property.
     * 
     * @return
     *     possible object is
     *     {@link NemusString }
     *     
     */
    public NemusString getRawdata() {
        return rawdata;
    }

    /**
     * Sets the value of the rawdata property.
     * 
     * @param value
     *     allowed object is
     *     {@link NemusString }
     *     
     */
    public void setRawdata(NemusString value) {
        this.rawdata = value;
    }

}
