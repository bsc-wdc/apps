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
package terasort.filesFilteredShared2;

import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;

public class Entry implements Externalizable, Comparable<Entry> {

    private long key = 0;
    private byte[] entry;

    /**
     * Empty Entry constructor.
     */
    public Entry() {
    }

    /**
     * Entry constructor
     * @param entry Byte array value
     * @param key Key
     */
    public Entry(byte[] entry, long key) {
        this.entry = entry;
        this.key = key;
    }

    /**
     * Key getter.
     * @return The Entry key.
     */
    public long getKey() {
        return key;
    }

    /**
     * Key setter.
     * @param key 
     */
    public void setKey(long key) {
        this.key = key;
    }

    /**
     * Entry value getter.
     * @return The Entry value.
     */
    public byte[] getEntry() {
        return entry;
    }

    /**
     * Entry value setter.
     * @param entry The entry value
     */
    public void setEntry(byte[] entry) {
        this.entry = entry;
    }

    /**
     * Pair setter
     * @param entry Entry value
     * @param key Key
     */
    public void setPair(byte[] entry, long key) {
        this.key = key;
        this.entry = entry;
    }

    /**
     * Entry comparison function.
     *
     * @param anotherInstance Entry to compare to
     * @return 0 if equal, 1 if current pair key is greater, -1 if
     * anotherInstance key is greater
     */
    @Override
    public int compareTo(Entry anotherInstance) {
        long anotherKey = anotherInstance.getKey();
        if (this.key == anotherKey) {
            return 0;
        } else if (this.key > anotherKey) {
            return 1;
        } else {
            return -1;
        }
    }

    /**
     * String representation of an Entry object
     * @return The string representation
     */
    public String toString() {
        String result = "Key: " + this.key + "\tEntry length: " + this.entry.length;
        return result;
    }

    /**
     * Externalizable serialization writer
     * @param oo Write destination
     * @throws IOException 
     */
    @Override
    public void writeExternal(ObjectOutput oo) throws IOException {
        oo.writeLong(key);
        oo.writeInt(entry.length);
        oo.write(entry);
    }

    /**
     * Externalizable serilization reader
     * @param oi Read source
     * @throws IOException
     * @throws ClassNotFoundException 
     */
    @Override
    public void readExternal(ObjectInput oi) throws IOException, ClassNotFoundException {
        this.key = oi.readLong();
        int length = oi.readInt();
        this.entry = new byte[length];
        oi.read(entry);
    }
}
