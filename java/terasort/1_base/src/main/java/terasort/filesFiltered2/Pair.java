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

public class Pair implements Serializable, Comparable<Pair> {

    private long key = 0;
    private byte[] entry;

    public Pair() {
    }

    public Pair(byte[] entry, long key) {
        this.entry = entry;
        this.key = key;
    }

    public long getKey() {
        return key;
    }

    public void setKey(long key) {
        this.key = key;
    }

    public byte[] getEntry() {
        return entry;
    }

    public void setEntry(byte[] entry) {
        this.entry = entry;
    }

    public void setPair(byte[] entry, long key) {
        this.key = key;
        this.entry = entry;
    }

    @Override
    public int compareTo(Pair anotherInstance) {
        long anotherKey = anotherInstance.getKey();
        if (this.key == anotherKey) {
            return 0;
        } else if (this.key > anotherKey) {
            return 1;
        } else {
            return -1;
        }
    }

    public String toString() {
        String result = "Key: " + this.key + "\tEntry length: " + this.entry.length;
        return result;
    }
}
