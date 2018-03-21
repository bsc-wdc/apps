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

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for runBlast2Gene complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="runBlast2Gene">
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;sequence>
 *         &lt;element name="blastResult" type="{http://genedetect.core}BLAST-Text" minOccurs="0"/>
 *         &lt;element name="params" type="{http://genedetect.core}RunBlast2GeneSecondaryParameters" minOccurs="0"/>
 *         &lt;element name="db" type="{http://genedetect.core}database" minOccurs="0"/>
 *       &lt;/sequence>
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "runBlast2Gene", propOrder = {
    "blastResult",
    "params",
    "db"
})
public class RunBlast2Gene {

    protected BLASTText blastResult;
    protected RunBlast2GeneSecondaryParameters params;
    protected Database db;

    /**
     * Gets the value of the blastResult property.
     * 
     * @return
     *     possible object is
     *     {@link BLASTText }
     *     
     */
    public BLASTText getBlastResult() {
        return blastResult;
    }

    /**
     * Sets the value of the blastResult property.
     * 
     * @param value
     *     allowed object is
     *     {@link BLASTText }
     *     
     */
    public void setBlastResult(BLASTText value) {
        this.blastResult = value;
    }

    /**
     * Gets the value of the params property.
     * 
     * @return
     *     possible object is
     *     {@link RunBlast2GeneSecondaryParameters }
     *     
     */
    public RunBlast2GeneSecondaryParameters getParams() {
        return params;
    }

    /**
     * Sets the value of the params property.
     * 
     * @param value
     *     allowed object is
     *     {@link RunBlast2GeneSecondaryParameters }
     *     
     */
    public void setParams(RunBlast2GeneSecondaryParameters value) {
        this.params = value;
    }

    /**
     * Gets the value of the db property.
     * 
     * @return
     *     possible object is
     *     {@link Database }
     *     
     */
    public Database getDb() {
        return db;
    }

    /**
     * Sets the value of the db property.
     * 
     * @param value
     *     allowed object is
     *     {@link Database }
     *     
     */
    public void setDb(Database value) {
        this.db = value;
    }

}
