/*
 *  Copyright 2002-2015 Barcelona Supercomputing Center (www.bsc.es)
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

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
