/*
 *  Copyright 2002-2016 Barcelona Supercomputing Center (www.bsc.es)
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
package terasort.filesFiltered2;

import java.io.Serializable;

public class Range implements Serializable {

    private long start;
    private long end;

    /* Default constructor */
    public Range() {
    }

    public Range(long start, long end) {
        this.start = start;
        this.end = end;
    }

    public void setStart(long start) {
        this.start = start;
    }

    public long getStart() {
        return this.start;
    }

    public void setEnd(long end) {
        this.end = end;
    }

    public long getEnd() {
        return this.end;
    }

    @Override
    public boolean equals(Object other) {
        if (other == null) {
            return false;
        }
        if (other == this) {
            return true;
        }
        if (!(other instanceof Range)) {
            return false;
        }

        Range otherMyClass = (Range) other;
        long s = otherMyClass.getStart();
        long e = otherMyClass.getEnd();
        if ((s != this.start) || (e != this.end)) {
            return false;
        } else {
            return true;
        }
    }

//    @Override
//    public String toString() {
//    	String result = "(" + this.start + ", " + this.end + ")";
//        return result;
//    }
}
