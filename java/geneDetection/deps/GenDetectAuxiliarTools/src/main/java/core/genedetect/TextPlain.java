
package core.genedetect;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlSeeAlso;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for text-plain complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="text-plain">
 *   &lt;complexContent>
 *     &lt;extension base="{http://genedetect.core}NemusObject">
 *       &lt;sequence>
 *         &lt;element name="content" type="{http://genedetect.core}NemusString" minOccurs="0"/>
 *       &lt;/sequence>
 *     &lt;/extension>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "text-plain", propOrder = {
    "content"
})
@XmlSeeAlso({
    TextFormatted.class
})
public class TextPlain
    extends NemusObject
{

    protected NemusString content;

    /**
     * Gets the value of the content property.
     * 
     * @return
     *     possible object is
     *     {@link NemusString }
     *     
     */
    public NemusString getContent() {
        return content;
    }

    /**
     * Sets the value of the content property.
     * 
     * @param value
     *     allowed object is
     *     {@link NemusString }
     *     
     */
    public void setContent(NemusString value) {
        this.content = value;
    }

}
