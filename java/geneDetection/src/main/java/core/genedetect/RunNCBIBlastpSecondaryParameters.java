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
 * <p>Java class for RunNCBIBlastpSecondaryParameters complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="RunNCBIBlastpSecondaryParameters">
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;all>
 *         &lt;element name="alignments" type="{http://www.w3.org/2001/XMLSchema}int" minOccurs="0"/>
 *         &lt;element name="database" type="{http://genedetect.core}database" minOccurs="0"/>
 *         &lt;element name="dropoff" type="{http://www.w3.org/2001/XMLSchema}int" minOccurs="0"/>
 *         &lt;element name="expected_threshold" type="{http://www.w3.org/2001/XMLSchema}float" minOccurs="0"/>
 *         &lt;element name="extendgap" type="{http://www.w3.org/2001/XMLSchema}int" minOccurs="0"/>
 *         &lt;element name="filter" type="{http://genedetect.core}filter" minOccurs="0"/>
 *         &lt;element name="gapalign" type="{http://genedetect.core}gapalign" minOccurs="0"/>
 *         &lt;element name="matrix" type="{http://genedetect.core}matrix" minOccurs="0"/>
 *         &lt;element name="opengap" type="{http://www.w3.org/2001/XMLSchema}int" minOccurs="0"/>
 *         &lt;element name="scores" type="{http://www.w3.org/2001/XMLSchema}int" minOccurs="0"/>
 *       &lt;/all>
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "RunNCBIBlastpSecondaryParameters", propOrder = {

})
public class RunNCBIBlastpSecondaryParameters {

    @XmlElement(defaultValue = "15")
    protected Integer alignments;
    @XmlElement(defaultValue = "UniProt")
    protected Database database;
    @XmlElement(defaultValue = "0")
    protected Integer dropoff;
    @XmlElement(name = "expected_threshold", defaultValue = "10.0")
    protected Float expectedThreshold;
    @XmlElement(defaultValue = "2")
    protected Integer extendgap;
    @XmlElement(defaultValue = "false")
    protected Filter filter;
    @XmlElement(defaultValue = "true")
    protected Gapalign gapalign;
    @XmlElement(defaultValue = "BLOSUM62")
    protected Matrix matrix;
    @XmlElement(defaultValue = "11")
    protected Integer opengap;
    @XmlElement(defaultValue = "25")
    protected Integer scores;

    /**
     * Gets the value of the alignments property.
     * 
     * @return
     *     possible object is
     *     {@link Integer }
     *     
     */
    public Integer getAlignments() {
        return alignments;
    }

    /**
     * Sets the value of the alignments property.
     * 
     * @param value
     *     allowed object is
     *     {@link Integer }
     *     
     */
    public void setAlignments(Integer value) {
        this.alignments = value;
    }

    /**
     * Gets the value of the database property.
     * 
     * @return
     *     possible object is
     *     {@link Database }
     *     
     */
    public Database getDatabase() {
        return database;
    }

    /**
     * Sets the value of the database property.
     * 
     * @param value
     *     allowed object is
     *     {@link Database }
     *     
     */
    public void setDatabase(Database value) {
        this.database = value;
    }

    /**
     * Gets the value of the dropoff property.
     * 
     * @return
     *     possible object is
     *     {@link Integer }
     *     
     */
    public Integer getDropoff() {
        return dropoff;
    }

    /**
     * Sets the value of the dropoff property.
     * 
     * @param value
     *     allowed object is
     *     {@link Integer }
     *     
     */
    public void setDropoff(Integer value) {
        this.dropoff = value;
    }

    /**
     * Gets the value of the expectedThreshold property.
     * 
     * @return
     *     possible object is
     *     {@link Float }
     *     
     */
    public Float getExpectedThreshold() {
        return expectedThreshold;
    }

    /**
     * Sets the value of the expectedThreshold property.
     * 
     * @param value
     *     allowed object is
     *     {@link Float }
     *     
     */
    public void setExpectedThreshold(Float value) {
        this.expectedThreshold = value;
    }

    /**
     * Gets the value of the extendgap property.
     * 
     * @return
     *     possible object is
     *     {@link Integer }
     *     
     */
    public Integer getExtendgap() {
        return extendgap;
    }

    /**
     * Sets the value of the extendgap property.
     * 
     * @param value
     *     allowed object is
     *     {@link Integer }
     *     
     */
    public void setExtendgap(Integer value) {
        this.extendgap = value;
    }

    /**
     * Gets the value of the filter property.
     * 
     * @return
     *     possible object is
     *     {@link Filter }
     *     
     */
    public Filter getFilter() {
        return filter;
    }

    /**
     * Sets the value of the filter property.
     * 
     * @param value
     *     allowed object is
     *     {@link Filter }
     *     
     */
    public void setFilter(Filter value) {
        this.filter = value;
    }

    /**
     * Gets the value of the gapalign property.
     * 
     * @return
     *     possible object is
     *     {@link Gapalign }
     *     
     */
    public Gapalign getGapalign() {
        return gapalign;
    }

    /**
     * Sets the value of the gapalign property.
     * 
     * @param value
     *     allowed object is
     *     {@link Gapalign }
     *     
     */
    public void setGapalign(Gapalign value) {
        this.gapalign = value;
    }

    /**
     * Gets the value of the matrix property.
     * 
     * @return
     *     possible object is
     *     {@link Matrix }
     *     
     */
    public Matrix getMatrix() {
        return matrix;
    }

    /**
     * Sets the value of the matrix property.
     * 
     * @param value
     *     allowed object is
     *     {@link Matrix }
     *     
     */
    public void setMatrix(Matrix value) {
        this.matrix = value;
    }

    /**
     * Gets the value of the opengap property.
     * 
     * @return
     *     possible object is
     *     {@link Integer }
     *     
     */
    public Integer getOpengap() {
        return opengap;
    }

    /**
     * Sets the value of the opengap property.
     * 
     * @param value
     *     allowed object is
     *     {@link Integer }
     *     
     */
    public void setOpengap(Integer value) {
        this.opengap = value;
    }

    /**
     * Gets the value of the scores property.
     * 
     * @return
     *     possible object is
     *     {@link Integer }
     *     
     */
    public Integer getScores() {
        return scores;
    }

    /**
     * Sets the value of the scores property.
     * 
     * @param value
     *     allowed object is
     *     {@link Integer }
     *     
     */
    public void setScores(Integer value) {
        this.scores = value;
    }

}
