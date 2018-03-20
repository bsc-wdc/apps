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
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.util.ArrayList;
import java.util.Collections;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.Paths;

public class Fragment implements Externalizable {

    private ArrayList<Entry> fragment = new ArrayList<Entry>();

    private int key_len = 10;
    private int value_len = 90;
    private int record_len = key_len + value_len;

    /**
     * Default constructor.
     */
    public Fragment() {
    }

    /**
     * Constructor from Pair arraylist.
     *
     * @param f Content for the fragment
     */
    public Fragment(ArrayList<Entry> f) {
        this.fragment = f;
    }

    /**
     * Constructor from file path. 
     * Files must follow the structure generated with Spark's Teragen application. 
     * Restriction 10 bytes for the key and 90 bytes for the value. 
     * Follows the reconstruction of the file as Spark's Terasort.
     *
     * @param filePath Content for the fragment
     */
    public Fragment(String filePath) {
        byte[] data = fastReadFile(filePath);
        long min = Long.MAX_VALUE;
        long max = Long.MIN_VALUE;
        for (int i = 0; i < data.length; i += record_len) {
            byte[] r = new byte[record_len];
            int pos = 0;
            for (int j = i; j < i + record_len; j++) {
                r[pos] = data[j];
                pos++;
            }
            byte[] realKeyEntry = new byte[8];
            realKeyEntry[0] = 0;
            realKeyEntry[1] = r[0];
            realKeyEntry[2] = r[1];
            realKeyEntry[3] = r[2];
            realKeyEntry[4] = r[3];
            realKeyEntry[5] = r[4];
            realKeyEntry[6] = r[5];
            realKeyEntry[7] = r[6];
            long eKey = Utils.byte2long(realKeyEntry);
            //System.out.println("Key: " + eKey);
            //System.out.println("Record: " + new String(r));
            fragment.add(new Entry(r, eKey));
        }
    }

    public Fragment(String filePath, long startPos, long endPos) {
        byte[] data = fastReadFileBlock(filePath, startPos, endPos);
        for (int i = 0; i < data.length; i += record_len) {
            byte[] r = new byte[record_len];
            int pos = 0;
            for (int j = i; j < i + record_len; j++) {
                r[pos] = data[j];
                pos++;
            }
            byte[] realKeyEntry = new byte[8];
            realKeyEntry[0] = 0;
            realKeyEntry[1] = r[0];
            realKeyEntry[2] = r[1];
            realKeyEntry[3] = r[2];
            realKeyEntry[4] = r[3];
            realKeyEntry[5] = r[4];
            realKeyEntry[6] = r[5];
            realKeyEntry[7] = r[6];
            long eKey = Utils.byte2long(realKeyEntry);
            //System.out.println("Key: " + eKey);
            //System.out.println("Record: " + new String(r));
            fragment.add(new Entry(r, eKey));
        }
    }

    /**
     * Retrieve a new fragment with the keys between the range of startValue and
     * endValue.
     *
     * @param startValue Key range start
     * @param endValue Key range end
     * @return A Fragment with a subset of entries
     */
    public Fragment filter(long startValue, long endValue) {
        ArrayList<Entry> filteredFragment = new ArrayList<Entry>();
        for (Entry p : this.fragment) {
            if (p.getKey() >= startValue && p.getKey() < endValue) {
                filteredFragment.add(p);
            }
        }
        //for(Entry p: filteredFragment){
        //    System.out.println("Filtered Key: "+ p.getKey());
        //    System.out.println("Filtered Record: "+ new String(p.getEntry()));
        //}
        return new Fragment(filteredFragment);
    }

    /**
     * Retrieve the fragment size.
     *
     * @return Fragment size
     */
    public long size() {
        return this.fragment.size();
    }

    /**
     * Divide the Fragment into n parts.
     * @param parts Number of parts to divide the Fragment
     * @return An array with the Fragment parts
     */
    public Fragment[] partition(int parts) {
        int numElems = this.fragment.size() / parts;
        int start = 0;
        Fragment[] p = new Fragment[parts];
        for (int i = 0; i < parts; i++) {
            p[i] = new Fragment(new ArrayList<Entry>(this.fragment.subList(start, start + numElems)));
            start += numElems;
        }
        return p;
    }

    /**
     * Fast file reading function
     * @param filePath File to read
     * @return The contents of the file as byte array
     */
    private byte[] fastReadFile(String filePath) {
        byte[] result = null;
        try {
            File file = new File(filePath);
            long fileSize = file.length();

            Path path = Paths.get(filePath);
            FileChannel fileChannel = FileChannel.open(path);

            ByteBuffer buffer;
            buffer = ByteBuffer.allocate((int) fileSize);
            int noOfBytesRead = fileChannel.read(buffer);
            fileChannel.close();
            result = buffer.array();
        } catch (FileNotFoundException ex) {
            System.out.println("[ERROR] Exception " + ex);
        } catch (IOException ex) {
            System.out.println("[ERROR] Exception " + ex);
        }

        return result;

    }

    /**
     * Fast file reading function for a particular block
     * @param filePath File to read
     * @param startPos Start position
     * @param endPos End position
     * @return The contents of the file between start and end as byte array
     */
    private byte[] fastReadFileBlock(String filePath, long startPos, long endPos) {
        byte[] result = null;
        long blockSize = endPos - startPos; // bytes to read
        try {
            File file = new File(filePath);
            //long fileSize = file.length();

            Path path = Paths.get(filePath);
            FileChannel fileChannel = FileChannel.open(path);

            ByteBuffer buffer;
            buffer = ByteBuffer.allocate((int) blockSize);
            int noOfBytesRead = fileChannel.read(buffer, startPos);
            fileChannel.close();
            result = buffer.array();
        } catch (FileNotFoundException ex) {
            System.out.println("[ERROR] Exception " + ex);
        } catch (IOException ex) {
            System.out.println("[ERROR] Exception " + ex);
        }

        return result;

    }

    /**
     * Extend the current fragment with the contents of another fragment keeping
     * it sorted.
     *
     * @param f The fragment to merge with this one.
     */
    public void put(Fragment f) {
        this.fragment.addAll(f.getFragment());
        Collections.sort(this.fragment);
        // Ineficient
        //for (int index1 = 0, index2 = 0; index2 < f.size(); index1++) {
        //    if (index1 == this.fragment.size() || this.fragment.get(index1).getKey() > f.fragment.get(index2).getKey()) {
        //        this.fragment.add(index1, f.fragment.get(index2++));
        //    }
        //}
    }

    /**
     * Returns the Pair arraylist (fragment contents). This arraylist contains
     * one object Pair per key.
     *
     * @return The Pair arraylist.
     */
    public ArrayList<Entry> getFragment() {
        return this.fragment;
    }

    /**
     * Sort the fragment. Uses compare from Pair class.
     *
     * @return The fragment sorted.
     */
    public Fragment sort() {
        Collections.sort(this.fragment);
        return this;
    }

    /**
     * Fast file write function
     * @param filePath File to read
     * @return The number of entries written
     */
    public int fastSave(String filePath) {
        int count = 0;
        int numEntries = this.fragment.size();
        // this option takes about 5 seconds every 500 Mb
        int resultSize = numEntries * this.record_len;
        byte[] result = new byte[resultSize];
        int pos = 0;
        for (int i = 0; i < numEntries; ++i) {
            byte[] e = this.fragment.get(i).getEntry();
            for (int j = 0; j < this.record_len; ++j) {
                result[pos + j] = e[j];
            }
            pos += this.record_len;
            count++;
        }
        try {
            FileOutputStream fos = new FileOutputStream(filePath);
            FileChannel fileChannel = fos.getChannel();
            ByteBuffer buffer = ByteBuffer.wrap(result);
            fileChannel.write(buffer);
            fileChannel.close();
            fos.close();
        } catch (FileNotFoundException ex) {
            System.out.println("[ERROR] Exception " + ex);
        } catch (IOException ex) {
            System.out.println("[ERROR] Exception " + ex);
        }
        return count;
    }

    /**
     * Externalizable serialization writer
     * @param oo Write destination
     * @throws IOException 
     */
    @Override
    public void writeExternal(ObjectOutput oo) throws IOException {
        oo.writeInt(key_len);
        oo.writeInt(this.value_len);
        oo.writeInt(this.record_len);
        oo.writeObject(this.fragment);
    }

    /**
     * Externalizable serilization reader
     * @param oi Read source
     * @throws IOException
     * @throws ClassNotFoundException 
     */
    @Override
    public void readExternal(ObjectInput oi) throws IOException, ClassNotFoundException {
        this.key_len = oi.readInt();
        this.value_len = oi.readInt();
        this.record_len = oi.readInt();
        this.fragment = (ArrayList<Entry>) oi.readObject();

    }
}
