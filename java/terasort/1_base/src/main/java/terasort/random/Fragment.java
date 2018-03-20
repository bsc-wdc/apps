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
package terasort.random;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

public class Fragment implements Serializable {

    private ArrayList<Pair> fragment = new ArrayList<>();

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
     * Random dataset fragment generator.
     *
     * @param numKeys Number of entries per fragment
     * @param uniqueKeys Number of unique keys (range)
     * @param keyLength Key length
     * @param uniqueValues Number of unique values (range)
     * @param valueLength Value length
     * @param randomSeed Seed to be used
     */
    public Fragment(int numKeys, int uniqueKeys, int keyLength, int uniqueValues, int valueLength, long randomSeed) {
        Random generator = new Random(randomSeed);
        for (int i = 0; i < numKeys; ++i) {
            int key = (int) (generator.nextDouble() * uniqueKeys);
            int value = (int) (generator.nextDouble() * uniqueValues);
            String skey = String.valueOf(key);
            skey = String.format("%" + keyLength + "s", skey).replace(' ', '0');
            String svalue = String.valueOf(value);
            svalue = String.format("%" + valueLength + "s", svalue).replace(' ', '0');
            this.fragment.add(new Pair(skey, svalue));
        }
    }

    /**
     * Constructor from file path. Each file must have a key and a value per
     * line separated with a space
     *
     * @param filePath Content for the fragment
     */
    public Fragment(String filePath) {
        File file = new File(filePath);

        FileReader fr = null;
        BufferedReader br = null;
        try {
            fr = new FileReader(file);
            br = new BufferedReader(fr);
            String line;
            while ((line = br.readLine()) != null) {
                String[] keyValuePair = line.split(" ");
                this.fragment.add(new Pair(keyValuePair[0], keyValuePair[1]));
            }
        } catch (Exception e) {
            System.err.println("ERROR: Cannot retrieve values from " + file.getName());
            e.printStackTrace();
        } finally {
            if (br != null) {
                try {
                    br.close();
                } catch (Exception e) {
                    System.err.println("ERROR: Cannot close buffered reader on file " + file.getName());
                    e.printStackTrace();
                }
            }
            if (fr != null) {
                try {
                    fr.close();
                } catch (Exception e) {
                    System.err.println("ERROR: Cannot close file reader on file " + file.getName());
                    e.printStackTrace();
                }
            }
        }
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
    public Fragment filter(int startValue, int endValue) {
        ArrayList<Pair> filtered = new ArrayList<Pair>();
        for (int i = 0; i < this.fragment.size(); ++i) {
            int key = Integer.valueOf(this.fragment.get(i).getKey());
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
    public String[] getValues() {
        int numKeys = this.fragment.size();
        String[] values = new String[numKeys];
        for (int i = 0; i < numKeys; ++i) {
            values[i] = this.fragment.get(i).getValue();
        }
        return values;
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
        for (Pair p : this.fragment) {
            result = result + Integer.valueOf(p.getKey()) + "\t";
        }
        result += "\nVals = \t";
        for (Pair p : this.fragment) {
            result = result + Integer.valueOf(p.getValue()) + "\t";
        }
        result += "\n";
        return result;
    }
     */
}
