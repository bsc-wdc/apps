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

package npb.nasep;

import java.io.Serializable;


public class EPValues implements Serializable {
	/**
	 * Serial version out of Runtime cannot be 1L nor 2L
	 */
	private static final long serialVersionUID = 3L;

    private double sx;
    private double sy;
    private double[] q;
    private double gc;

    public void setSx(double sx){
        this.sx=sx;
    }

    public void setSy(double sy){
        this.sy=sy;
    }

    public void setQ(double[] q){
        this.q=q;
    }

    public void setGc(double gc){
        this.gc=gc;
    }

    public double getSx(){
        return this.sx;
    }

    public double getSy(){
        return this.sy;
    }

    public double[] getQ(){
        return this.q;
    }

    public double getGc(){
        return this.gc;
    }
}
