
package core.genedetect;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlSeeAlso;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for Annotation complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="Annotation">
 *   &lt;complexContent>
 *     &lt;extension base="{http://genedetect.core}NemusObject">
 *       &lt;sequence>
 *         &lt;element name="End" type="{http://genedetect.core}NemusInteger" minOccurs="0"/>
 *         &lt;element name="Start" type="{http://genedetect.core}NemusInteger" minOccurs="0"/>
 *       &lt;/sequence>
 *     &lt;/extension>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "Annotation", propOrder = {
    "end",
    "start"
})
@XmlSeeAlso({
    BL2GAnnotation.class
})
public class Annotation
    extends NemusObject
{

    @XmlElement(name = "End")
    protected NemusInteger end;
    @XmlElement(name = "Start")
    protected NemusInteger start;

    /**
     * Gets the value of the end property.
     * 
     * @return
     *     possible object is
     *     {@link NemusInteger }
     *     
     */
    public NemusInteger getEnd() {
        return end;
    }

    /**
     * Sets the value of the end property.
     * 
     * @param value
     *     allowed object is
     *     {@link NemusInteger }
     *     
     */
    public void setEnd(NemusInteger value) {
        this.end = value;
    }

    /**
     * Gets the value of the start property.
     * 
     * @return
     *     possible object is
     *     {@link NemusInteger }
     *     
     */
    public NemusInteger getStart() {
        return start;
    }

    /**
     * Sets the value of the start property.
     * 
     * @param value
     *     allowed object is
     *     {@link NemusInteger }
     *     
     */
    public void setStart(NemusInteger value) {
        this.start = value;
    }

}
