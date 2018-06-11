
package core.genedetect;

import javax.xml.bind.annotation.XmlEnum;
import javax.xml.bind.annotation.XmlEnumValue;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for filter.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="filter">
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string">
 *     &lt;enumeration value="false"/>
 *     &lt;enumeration value="true"/>
 *   &lt;/restriction>
 * &lt;/simpleType>
 * </pre>
 * 
 */
@XmlType(name = "filter")
@XmlEnum
public enum Filter {

    @XmlEnumValue("false")
    FALSE("false"),
    @XmlEnumValue("true")
    TRUE("true");
    private final String value;

    Filter(String v) {
        value = v;
    }

    public String value() {
        return value;
    }

    public static Filter fromValue(String v) {
        for (Filter c: Filter.values()) {
            if (c.value.equals(v)) {
                return c;
            }
        }
        throw new IllegalArgumentException(v);
    }

}
