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
package terasort.files;

import java.io.Serializable;

public class Pair implements Serializable, Comparable<Pair> {

    private String key;
    private byte[] value;

    /**
     * Default empty constructor.
     */
    public Pair() {
    }

    /**
     * Pair constructor from key and value.
     *
     * @param key Pair key
     * @param value Pair value
     */
    public Pair(String key, byte[] value) {
        this.key = key;
        this.value = value;
    }

    /**
     * Key getter.
     *
     * @return The pair key
     */
    public String getKey() {
        return key;
    }

    /**
     * Key setter.
     *
     * @param key The pair key
     */
    public void setKey(String key) {
        this.key = key;
    }

    /**
     * Value getter
     *
     * @return The pair value
     */
    public byte[] getValue() {
        return value;
    }

    /**
     * Value setter.
     *
     * @param value The pair value
     */
    public void setValue(byte[] value) {
        this.value = value;
    }

    /**
     * Pair setter.
     *
     * @param key The pair key
     * @param value The pair value
     */
    public void setPair(String key, byte[] value) {
        this.key = key;
        this.value = value;
    }

    /**
     * Pair key comparison function.
     *
     * @param anotherInstance Pair to compare to
     * @return 0 if equal, 1 if current pair key is greater, -1 if
     * anotherInstance key is greater
     */
    @Override
    public int compareTo(Pair anotherInstance) {
        long key = Long.valueOf(this.key);
        long anotherKey = Long.valueOf(anotherInstance.getKey());
        if (key == anotherKey) {
            return 0;
        } else if (key > anotherKey) {
            return 1;
        } else {
            return -1;
        }
    }

    /**
     * String representation of a pair.
     *
     * @return Pair as string representation
     */
    public String toString() {
        String result = "Key: " + this.key + "\tValue: " + this.value;
        return result;
    }
}
