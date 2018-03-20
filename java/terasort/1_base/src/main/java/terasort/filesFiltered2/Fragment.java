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

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.Paths;

public class Fragment implements Serializable {

    private ArrayList<Pair> fragment = new ArrayList<Pair>();
    private Range range;

    private int key_len = 10;
    private int value_len = 90;
    private int record_len = key_len + value_len;

    /* Default constructor */
    public Fragment() {
    }

    /* Constructor from Pair arraylist. */
    public Fragment(ArrayList<Pair> f, Range range) {
        this.fragment = f;
        this.range = range;
    }

    public Fragment(String filePath, long start, long end) {
        byte[] data = fastReadFile(filePath);

        this.range = new Range(start, end);

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

            if (eKey >= start && eKey < end) {
                fragment.add(new Pair(r, eKey));
            }
        }
    }

    public Range getRange() {
        return this.range;
    }

    public void setRange(Range range) {
        this.range = range;
    }

//    /**
//     * Read the given binary file, and return its contents as a byte array. 
//     * @param aInputFileName File to read.
//     * @return byte[] file content.
//     */
//    private byte[] readFile(String aInputFileName){
//    	File file = new File(aInputFileName);
//    	byte[] result = new byte[(int)file.length()];
//    	try {
//            InputStream input = null;
//            try {
//                int totalBytesRead = 0;
//                input = new BufferedInputStream(new FileInputStream(file));
//                while(totalBytesRead < result.length){
//                    int bytesRemaining = result.length - totalBytesRead;
//                    //input.read() returns -1, 0, or more :
//                    int bytesRead = input.read(result, totalBytesRead, bytesRemaining); 
//                    if (bytesRead > 0){
//                        totalBytesRead = totalBytesRead + bytesRead;
//                    }
//                }
//            }
//            finally {
//                input.close();
//            }
//    	}
//    	catch (FileNotFoundException ex) {
//    		System.out.println("Exception " + ex);
//    	}
//    	catch (IOException ex) {
//    		System.out.println("Exception " + ex);
//    	}
//    	return result;
//    }
    
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
            System.out.println("Exception " + ex);
        } catch (IOException ex) {
            System.out.println("Exception " + ex);
        }

        return result;

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
     * Returns the Pair arraylist (fragment contents). This arraylist contains
     * one object Pair per key.
     *
     * @return The Pair arraylist.
     */
    public ArrayList<Pair> getFragment() {
        return this.fragment;
    }

    public void setFragment(ArrayList<Pair> f) {
        this.fragment = f;
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

//    // this option takes about 190 seconds every 500 Mb    
//    /**
//     * Save the fragment contents to a file.
//     * @param filePath File where to store the fragment.
//     * @return Integer - The amount of elements stored.
//     */
//    public int save(String filePath){
//    	//int key_len = 10;
//        //int value_len = 90;
//        //int record_len = key_len + value_len;
//
//        String firstKey = String.valueOf(this.fragment.get(0).getKey());
//
//        int recordNumber = 0;
//        
//
//        for (int i=0; i<fragment.size(); ++i){
//            byte[] toWrite = this.fragment.get(i).getEntry();
//            write(toWrite, filePath + "/" + firstKey);
//            recordNumber++;
//        }
//    	
//    	return recordNumber;
//    }
//    
//    /**
//     * Write a byte array to the given file.
//     * (Writing binary data is significantly simpler than reading it)
//     * @param aInput byte[] to write.
//     * @param aOutputFileName Filename where to write the input byte array.
//     */
//    void write(byte[] aInput, String aOutputFileName){
//    	try {
//            OutputStream output = null;
//            try {
//                output = new BufferedOutputStream(new FileOutputStream(aOutputFileName, true));
//                output.write(aInput);
//            }
//            finally {
//                output.close();
//            }
//    	}
//    	catch(FileNotFoundException ex){
//            System.out.println("Exception " + ex);
//    	}
//    	catch(IOException ex){
//            System.out.println("Exception " + ex);
//    	}
//    }
    
    public int fastSave(String filePath) {
        if (this.fragment.size() == 0) {
            return 0;
        } else {
            String firstKey = String.valueOf(this.fragment.get(0).getKey());
            String file = filePath + "/" + firstKey;
            int count = 0;

            int numEntries = this.fragment.size();

            // this option takes about 5 seconds every 500 Mb
            int resultSize = numEntries * this.record_len;
            byte[] result = new byte[resultSize];
            for (int i = 0; i < numEntries; ++i) {
                byte[] e = this.fragment.get(i).getEntry();
                for (int j = 0; j < this.record_len; ++j) {
                    result[i + j] = e[j];
                }
                count++;
            }
            try {
                FileOutputStream fos = new FileOutputStream(file);
                FileChannel fileChannel = fos.getChannel();
                ByteBuffer buffer = ByteBuffer.wrap(result);
                fileChannel.write(buffer);
                fileChannel.close();
                fos.close();
                /*
                // this option takes about 25 seconds every 500 Mb
                FileOutputStream fos = new FileOutputStream(file);
                FileChannel fileChannel = fos.getChannel();
                for (int i=0 ; i< numEntries; ++i){
                    byte[] e = this.fragment.get(i).getEntry();
                    ByteBuffer buffer = ByteBuffer.wrap(e);
                    fileChannel.write(buffer);
                    count++;
                }
                fileChannel.close();
                fos.close();	 
                */
            } catch (FileNotFoundException ex) {
                System.out.println("Exception " + ex);
            } catch (IOException ex) {
                System.out.println("Exception " + ex);
            }

            return count;
        }
    }


//    // Commented due for not to show its contents during execution.
//    @Override
//    public String toString() {
//    	String result = "";
//    	result += "Keys = \n";
//    	for (Pair p: this.fragment){
//    		result = result + Long.valueOf(p.getKey()) + "\t"; 
//    	}
//        return result;
//    }
}

