
package core.genedetect;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlType;
import javax.xml.bind.annotation.XmlValue;


/**
 * <p>Java class for NemusString complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="NemusString">
 *   &lt;simpleContent>
 *     &lt;extension base="&lt;http://www.w3.org/2001/XMLSchema>string">
 *       &lt;attribute name="nemusId" type="{http://www.w3.org/2001/XMLSchema}string" />
 *       &lt;attribute name="nemusNamespace" type="{http://www.w3.org/2001/XMLSchema}string" />
 *     &lt;/extension>
 *   &lt;/simpleContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "NemusString", propOrder = {
    "value"
})
public class NemusString {

    @XmlValue
    protected String value;
    @XmlAttribute(name = "nemusId")
    protected String nemusId;
    @XmlAttribute(name = "nemusNamespace")
    protected String nemusNamespace;

    /**
     * Gets the value of the value property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getValue() {
        return value;
    }

    /**
     * Sets the value of the value property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setValue(String value) {
        this.value = value;
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
