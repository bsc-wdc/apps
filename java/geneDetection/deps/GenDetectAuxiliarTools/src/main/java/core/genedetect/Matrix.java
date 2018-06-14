
package core.genedetect;

import javax.xml.bind.annotation.XmlEnum;
import javax.xml.bind.annotation.XmlEnumValue;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for matrix.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="matrix">
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string">
 *     &lt;enumeration value="PAM30"/>
 *     &lt;enumeration value="PAM70"/>
 *     &lt;enumeration value="BLOSUM45"/>
 *     &lt;enumeration value="BLOSUM62"/>
 *     &lt;enumeration value="BLOSUM80"/>
 *   &lt;/restriction>
 * &lt;/simpleType>
 * </pre>
 * 
 */
@XmlType(name = "matrix")
@XmlEnum
public enum Matrix {

    @XmlEnumValue("PAM30")
    PAM_30("PAM30"),
    @XmlEnumValue("PAM70")
    PAM_70("PAM70"),
    @XmlEnumValue("BLOSUM45")
    BLOSUM_45("BLOSUM45"),
    @XmlEnumValue("BLOSUM62")
    BLOSUM_62("BLOSUM62"),
    @XmlEnumValue("BLOSUM80")
    BLOSUM_80("BLOSUM80");
    private final String value;

    Matrix(String v) {
        value = v;
    }

    public String value() {
        return value;
    }

    public static Matrix fromValue(String v) {
        for (Matrix c: Matrix.values()) {
            if (c.value.equals(v)) {
                return c;
            }
        }
        throw new IllegalArgumentException(v);
    }

}
