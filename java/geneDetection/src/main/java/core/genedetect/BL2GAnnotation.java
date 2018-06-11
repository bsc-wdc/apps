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
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for BL2GAnnotation complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="BL2GAnnotation">
 *   &lt;complexContent>
 *     &lt;extension base="{http://genedetect.core}Annotation">
 *       &lt;sequence>
 *         &lt;element name="Coverage" type="{http://genedetect.core}NemusFloat" minOccurs="0"/>
 *         &lt;element name="db" type="{http://genedetect.core}NemusString" minOccurs="0"/>
 *         &lt;element name="Identity" type="{http://genedetect.core}NemusFloat" minOccurs="0"/>
 *         &lt;element name="ProtID" type="{http://genedetect.core}NemusString" minOccurs="0"/>
 *         &lt;element name="Strand" type="{http://genedetect.core}NemusString" minOccurs="0"/>
 *         &lt;element name="Type" type="{http://genedetect.core}NemusString" minOccurs="0"/>
 *       &lt;/sequence>
 *     &lt;/extension>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "BL2GAnnotation", propOrder = {
    "coverage",
    "db",
    "identity",
    "protID",
    "strand",
    "type"
})
public class BL2GAnnotation
    extends Annotation
{

    @XmlElement(name = "Coverage")
    protected NemusFloat coverage;
    protected NemusString db;
    @XmlElement(name = "Identity")
    protected NemusFloat identity;
    @XmlElement(name = "ProtID")
    protected NemusString protID;
    @XmlElement(name = "Strand")
    protected NemusString strand;
    @XmlElement(name = "Type")
    protected NemusString type;

    /**
     * Gets the value of the coverage property.
     * 
     * @return
     *     possible object is
     *     {@link NemusFloat }
     *     
     */
    public NemusFloat getCoverage() {
        return coverage;
    }

    /**
     * Sets the value of the coverage property.
     * 
     * @param value
     *     allowed object is
     *     {@link NemusFloat }
     *     
     */
    public void setCoverage(NemusFloat value) {
        this.coverage = value;
    }

    /**
     * Gets the value of the db property.
     * 
     * @return
     *     possible object is
     *     {@link NemusString }
     *     
     */
    public NemusString getDb() {
        return db;
    }

    /**
     * Sets the value of the db property.
     * 
     * @param value
     *     allowed object is
     *     {@link NemusString }
     *     
     */
    public void setDb(NemusString value) {
        this.db = value;
    }

    /**
     * Gets the value of the identity property.
     * 
     * @return
     *     possible object is
     *     {@link NemusFloat }
     *     
     */
    public NemusFloat getIdentity() {
        return identity;
    }

    /**
     * Sets the value of the identity property.
     * 
     * @param value
     *     allowed object is
     *     {@link NemusFloat }
     *     
     */
    public void setIdentity(NemusFloat value) {
        this.identity = value;
    }

    /**
     * Gets the value of the protID property.
     * 
     * @return
     *     possible object is
     *     {@link NemusString }
     *     
     */
    public NemusString getProtID() {
        return protID;
    }

    /**
     * Sets the value of the protID property.
     * 
     * @param value
     *     allowed object is
     *     {@link NemusString }
     *     
     */
    public void setProtID(NemusString value) {
        this.protID = value;
    }

    /**
     * Gets the value of the strand property.
     * 
     * @return
     *     possible object is
     *     {@link NemusString }
     *     
     */
    public NemusString getStrand() {
        return strand;
    }

    /**
     * Sets the value of the strand property.
     * 
     * @param value
     *     allowed object is
     *     {@link NemusString }
     *     
     */
    public void setStrand(NemusString value) {
        this.strand = value;
    }

    /**
     * Gets the value of the type property.
     * 
     * @return
     *     possible object is
     *     {@link NemusString }
     *     
     */
    public NemusString getType() {
        return type;
    }

    /**
     * Sets the value of the type property.
     * 
     * @param value
     *     allowed object is
     *     {@link NemusString }
     *     
     */
    public void setType(NemusString value) {
        this.type = value;
    }

}
