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
 * <p>Java class for CommentedNASequence complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="CommentedNASequence">
 *   &lt;complexContent>
 *     &lt;extension base="{http://genedetect.core}NucleotideSequence">
 *       &lt;sequence>
 *         &lt;element name="Description" type="{http://genedetect.core}NemusString" minOccurs="0"/>
 *       &lt;/sequence>
 *     &lt;/extension>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "CommentedNASequence", propOrder = {
    "description"
})
public class CommentedNASequence
    extends NucleotideSequence
{

    @XmlElement(name = "Description")
    protected NemusString description;

    /**
     * Gets the value of the description property.
     * 
     * @return
     *     possible object is
     *     {@link NemusString }
     *     
     */
    public NemusString getDescription() {
        return description;
    }

    /**
     * Sets the value of the description property.
     * 
     * @param value
     *     allowed object is
     *     {@link NemusString }
     *     
     */
    public void setDescription(NemusString value) {
        this.description = value;
    }

}
