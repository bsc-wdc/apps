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
 * <p>Java class for database.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="database">
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string">
 *     &lt;enumeration value="uniref90"/>
 *     &lt;enumeration value="TrEMBL"/>
 *     &lt;enumeration value="Swiss-Prot"/>
 *     &lt;enumeration value="UniProt"/>
 *     &lt;enumeration value="PDB"/>
 *     &lt;enumeration value="nr"/>
 *     &lt;enumeration value="RefSeq"/>
 *     &lt;enumeration value="RefSeq fungi"/>
 *     &lt;enumeration value="RefSeq invertebrate"/>
 *     &lt;enumeration value="RefSeq microbial"/>
 *     &lt;enumeration value="RefSeq mitochondrion"/>
 *     &lt;enumeration value="RefSeq plant"/>
 *     &lt;enumeration value="RefSeq plasmid"/>
 *     &lt;enumeration value="RefSeq plastid"/>
 *     &lt;enumeration value="RefSeq protozoa"/>
 *     &lt;enumeration value="RefSeq vertebrate_mammalian"/>
 *     &lt;enumeration value="RefSeq vertebrate_other"/>
 *     &lt;enumeration value="RefSeq viral"/>
 *   &lt;/restriction>
 * &lt;/simpleType>
 * </pre>
 * 
 */
@XmlType(name = "database")
@XmlEnum
public enum Database {

    @XmlEnumValue("uniref90")
    UNIREF_90("uniref90"),
    @XmlEnumValue("TrEMBL")
    TR_EMBL("TrEMBL"),
    @XmlEnumValue("Swiss-Prot")
    SWISS_PROT("Swiss-Prot"),
    @XmlEnumValue("UniProt")
    UNI_PROT("UniProt"),
    PDB("PDB"),
    @XmlEnumValue("nr")
    NR("nr"),
    @XmlEnumValue("RefSeq")
    REF_SEQ("RefSeq"),
    @XmlEnumValue("RefSeq fungi")
    REF_SEQ_FUNGI("RefSeq fungi"),
    @XmlEnumValue("RefSeq invertebrate")
    REF_SEQ_INVERTEBRATE("RefSeq invertebrate"),
    @XmlEnumValue("RefSeq microbial")
    REF_SEQ_MICROBIAL("RefSeq microbial"),
    @XmlEnumValue("RefSeq mitochondrion")
    REF_SEQ_MITOCHONDRION("RefSeq mitochondrion"),
    @XmlEnumValue("RefSeq plant")
    REF_SEQ_PLANT("RefSeq plant"),
    @XmlEnumValue("RefSeq plasmid")
    REF_SEQ_PLASMID("RefSeq plasmid"),
    @XmlEnumValue("RefSeq plastid")
    REF_SEQ_PLASTID("RefSeq plastid"),
    @XmlEnumValue("RefSeq protozoa")
    REF_SEQ_PROTOZOA("RefSeq protozoa"),
    @XmlEnumValue("RefSeq vertebrate_mammalian")
    REF_SEQ_VERTEBRATE_MAMMALIAN("RefSeq vertebrate_mammalian"),
    @XmlEnumValue("RefSeq vertebrate_other")
    REF_SEQ_VERTEBRATE_OTHER("RefSeq vertebrate_other"),
    @XmlEnumValue("RefSeq viral")
    REF_SEQ_VIRAL("RefSeq viral");
    private final String value;

    Database(String v) {
        value = v;
    }

    public String value() {
        return value;
    }

    public static Database fromValue(String v) {
        for (Database c: Database.values()) {
            if (c.value.equals(v)) {
                return c;
            }
        }
        throw new IllegalArgumentException(v);
    }

}
