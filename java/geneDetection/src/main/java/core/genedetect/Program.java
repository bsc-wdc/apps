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

import java.io.Serializable;

import javax.xml.bind.annotation.XmlEnum;
import javax.xml.bind.annotation.XmlEnumValue;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for program.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="program">
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string">
 *     &lt;enumeration value="blastp"/>
 *     &lt;enumeration value="blastn"/>
 *     &lt;enumeration value="blastx"/>
 *     &lt;enumeration value="tblastn"/>
 *     &lt;enumeration value="tblastx"/>
 *   &lt;/restriction>
 * &lt;/simpleType>
 * </pre>
 * 
 */
@XmlType(name = "program")
@XmlEnum
public enum Program implements Serializable {

    @XmlEnumValue("blastp")
    BLASTP("blastp"),
    @XmlEnumValue("blastn")
    BLASTN("blastn"),
    @XmlEnumValue("blastx")
    BLASTX("blastx"),
    @XmlEnumValue("tblastn")
    TBLASTN("tblastn"),
    @XmlEnumValue("tblastx")
    TBLASTX("tblastx");
    private final String value;

    Program(String v) {
        value = v;
    }

    public String value() {
        return value;
    }

    public static Program fromValue(String v) {
        for (Program c: Program.values()) {
            if (c.value.equals(v)) {
                return c;
            }
        }
        throw new IllegalArgumentException(v);
    }

}
