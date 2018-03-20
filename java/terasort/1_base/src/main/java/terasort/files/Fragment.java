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

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;

public class Fragment implements Serializable {

    private ArrayList<Pair> fragment = new ArrayList<Pair>();

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
    public Fragment(ArrayList<Pair> f) {
        this.fragment = f;
    }

    /**
     * Constructor from file path. Files must follow the structure generated
     * with Spark's Teragen application. Restriction 10 bytes for the key and 90
     * bytes for the value. Follows the reconstruction of the file as Spark's
     * Terasort.
     *
     * @param filePath Content for the fragment
     */
    public Fragment(String filePath) {
        byte[] fileContents = read(filePath);

        int key_len = 10;
        int value_len = 90;
        int record_len = key_len + value_len;

        for (int x = 0; x < fileContents.length; x += record_len) {

            String key;
            byte[] value;

            /* get Key */
            byte[] bkey = new byte[key_len];
            for (int i = 0; i < key_len; i++) {
                bkey[i] = fileContents[i + x];
            }

            byte[] aux = new byte[8];
            aux[0] = 0;
            aux[1] = bkey[0];
            aux[2] = bkey[1];
            aux[3] = bkey[2];
            aux[4] = bkey[3];
            aux[5] = bkey[4];
            aux[6] = bkey[5];
            aux[7] = bkey[6];
            key = String.valueOf(byte2long(aux));
            if (key.length() < 17) {
                int zeros = 17 - key.length();
                String zer = "";
                for (int z = 0; z < zeros; z++) {
                    zer += "0";
                }
                key = zer + key;
            }

            /* skip 2 bytes */
            //byte[] breakbytes = new byte[2];
            //breakbytes[0] = fileContents[x+ key_len];
            //breakbytes[1] = fileContents[x+ key_len+1];
            /* skip record number */
            //byte[] recordNumber = new byte[32];
            //int pos = key_len+2;
            //for (int i = 0; i<32; i++){
            //	recordNumber[i] = fileContents[x+ pos + i];
            //}
            /* skip break - 4 bytes */
            //byte[] breakdata = new byte[4];
            //breakdata[0] = fileContents[x+ 44];
            //breakdata[1] = fileContents[x+ 45];
            //breakdata[2] = fileContents[x+ 46];
            //breakdata[3] = fileContents[x+ 47];
            /* get value */
            byte[] randomNumber = new byte[48];
            int pos = 48;
            for (int i = 0; i < 48; i++) {
                randomNumber[i] = fileContents[x + pos + i];
            }
            value = randomNumber;
            /* skip last 4 bytes */
            //byte[] endbreakdata = new byte[4];
            //endbreakdata[0] = fileContents[x+ 96];
            //endbreakdata[1] = fileContents[x+ 97];
            //endbreakdata[2] = fileContents[x+ 98];
            //endbreakdata[3] = fileContents[x+ 99];
            //System.out.println("Key: " + key + "\t Value: " + value);
            this.fragment.add(new Pair(key, value));
        }
    }

    /**
     * Save the fragment contents to a file. Follows the structure of the
     * Terasort output file.
     *
     * @param filePath File where to store the fragment.
     * @return The amount of elements saved.
     */
    public int save(String filePath) {
        int key_len = 10;
        int value_len = 90;
        int record_len = key_len + value_len;
        //byte fill = this.hexStringToByte("ff");
        String firstKey = fragment.get(0).getKey();

        int recordNumber = 0;
        byte[] zero = this.intToByteArray(0);

        for (int i = 0; i < fragment.size(); ++i) {
            byte[] toWrite = new byte[record_len];
            String key = fragment.get(i).getKey();
            byte[] value = fragment.get(i).getValue();
            byte[] k = key.getBytes();
            for (int j = 0; j < key_len; ++j) {
                toWrite[j] = k[j];
            }
            /* add 2 bytes */
            toWrite[11] = this.hexStringToByte("00"); // fill;
            toWrite[12] = this.hexStringToByte("11"); // fill;

            /* skip record number */
            byte[] rn = this.intToByteArray(recordNumber);
            byte[] fullRN = new byte[32];
            System.arraycopy(zero, 0, fullRN, 0, 4);
            System.arraycopy(zero, 0, fullRN, 4, 4);
            System.arraycopy(zero, 0, fullRN, 8, 4);
            System.arraycopy(zero, 0, fullRN, 16, 4);
            System.arraycopy(zero, 0, fullRN, 24, 4);
            System.arraycopy(rn, 0, fullRN, 28, 4);
            for (int j = 0; j < 32; j++) {
                //toWrite[13 + j] = this.hexStringToByte("22"); // fill;
                toWrite[13 + j] = fullRN[j];
            }

            /* skip break - 4 bytes */
            toWrite[44] = this.hexStringToByte("88"); // fill;
            toWrite[45] = this.hexStringToByte("99"); // fill;
            toWrite[46] = this.hexStringToByte("aa"); // fill;
            toWrite[47] = this.hexStringToByte("bb"); // fill;

            /* set value */
            for (int j = 0; j < 48; j++) {
                toWrite[48 + j] = value[j];
            }

            /* skip break - 4 bytes */
            toWrite[96] = this.hexStringToByte("cc"); // fill;
            toWrite[97] = this.hexStringToByte("dd"); // fill;
            toWrite[98] = this.hexStringToByte("ee"); // fill;
            toWrite[99] = this.hexStringToByte("ff"); // fill;

            this.write(toWrite, filePath + "/" + firstKey);
        }

        return this.getCount();
    }

    /**
     * Returns the Pair arraylist (fragment contents). This arraylist contains
     * one object Pair per key.
     *
     * @return The Pair arraylist.
     */
    public ArrayList<Pair> getFragment() {
        return this.fragment;
    }

    /**
     * Set new content to this fragment.
     *
     * @param f New Pair arraylist to set.
     */
    public void setFragment(ArrayList<Pair> f) {
        this.fragment = f;
    }

    /**
     * Extend the current fragment with the contents of another fragment.
     *
     * @param f The fragment to merge with this one.
     */
    public void put(Fragment f) {
        this.fragment.addAll(f.getFragment());
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
        ArrayList<Pair> filtered = new ArrayList<Pair>();
        for (int i = 0; i < this.fragment.size(); ++i) {
            long key = Long.valueOf(this.fragment.get(i).getKey());
            if (key >= startValue && key < endValue) {
                filtered.add(this.fragment.get(i));
            }
        }
        return new Fragment(filtered);
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
     * Get all the keys of the fragment.
     *
     * @return All fragment keys as string array.
     */
    public String[] getKeys() {
        int numKeys = this.fragment.size();
        String[] keys = new String[numKeys];
        for (int i = 0; i < numKeys; ++i) {
            keys[i] = this.fragment.get(i).getKey();
        }
        return keys;
    }

    /**
     * Get the amount of entries K,V in the fragment
     *
     * @return The amount of pairs K,V of this fragment.
     */
    public int getCount() {
        return this.fragment.size();
    }

    /*
    // Commented due for not to show its contents during execution.
    public String toString() {
    	String result = "\n";
    	result += "Keys = \t";
    	for (Pair p: this.fragment){
    		result = result + Long.valueOf(p.getKey()) + "\n"; 
    	}
    	result += "\nVals = \t";
    	for (Pair p: this.fragment){
    		//result = result + Long.valueOf(p.getValue()) + "\t";
    		result = result + (String.valueOf(byte2long(p.getValue()))) + "\t";
    	}
    	result += "\n";
        return result;
    }
     */
    ///////////////////////////// AUXILIAR METHODS /////////////////////////////
    /**
     * Byte array to long converter.
     *
     * @param b Byte array to convert.
     * @return Long value from the byte array.
     */
    public long byte2long(byte[] b) {
        long result = -1;
        try {
            ByteArrayInputStream baos = new ByteArrayInputStream(b);
            DataInputStream dos = new DataInputStream(baos);
            result = dos.readLong();
            dos.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return result;
    }

    /**
     * Converts a integer into a 4 byte array.
     *
     * @param value Integer value.
     * @return byte[] 4 byte array value.
     */
    public byte[] intToByteArray(int value) {
        return new byte[]{
            (byte) (value >>> 24),
            (byte) (value >>> 16),
            (byte) (value >>> 8),
            (byte) value};
    }

    /**
     * Read the given binary file, and return its contents as a byte array.
     *
     * @param aInputFileName File to read.
     * @return byte[] file content.
     */
    byte[] read(String aInputFileName) {
        File file = new File(aInputFileName);
        byte[] result = new byte[(int) file.length()];
        try {
            InputStream input = null;
            try {
                int totalBytesRead = 0;
                input = new BufferedInputStream(new FileInputStream(file));
                while (totalBytesRead < result.length) {
                    int bytesRemaining = result.length - totalBytesRead;
                    //input.read() returns -1, 0, or more :
                    int bytesRead = input.read(result, totalBytesRead, bytesRemaining);
                    if (bytesRead > 0) {
                        totalBytesRead = totalBytesRead + bytesRead;
                    }
                }
            } finally {
                input.close();
            }
        } catch (FileNotFoundException ex) {
            System.out.println("Exception " + ex);
        } catch (IOException ex) {
            System.out.println("Exception " + ex);
        }
        return result;
    }

    /**
     * Write a byte array to the given file. (Writing binary data is
     * significantly simpler than reading it)
     *
     * @param aInput byte[] to write.
     * @param aOutputFileName Filename where to write the input byte array.
     */
    void write(byte[] aInput, String aOutputFileName) {
        try {
            OutputStream output = null;
            try {
                output = new BufferedOutputStream(new FileOutputStream(aOutputFileName, true));
                output.write(aInput);
            } finally {
                output.close();
            }
        } catch (FileNotFoundException ex) {
            System.out.println("Exception " + ex);
        } catch (IOException ex) {
            System.out.println("Exception " + ex);
        }
    }

    /**
     * Convert a hex string to byte
     *
     * @param data String to convert
     * @return Byte value
     */
    private byte hexStringToByte(String data) {
        return (byte) ((Character.digit(data.charAt(0), 16) << 4)
                + Character.digit(data.charAt(1), 16));
    }

}
