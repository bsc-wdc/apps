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

public class SortImpl {

    /**
     * Generate a Fragment object from the contents of a file
     *
     * @param numKeys Number of entries in the fragment
     * @param uniqueKeys Amount of unique keys
     * @param keyLength Key length
     * @param uniqueValues Amount of unique values
     * @param valueLength Value length
     * @param randomSeed Seed to use to generate the entries
     * @return A random generated Fragment
     */
    public static Fragment generateFragment(int numKeys, int uniqueKeys, int keyLength, int uniqueValues,
            int valueLength, long randomSeed) {
        Fragment fragment = new Fragment(numKeys, uniqueKeys, keyLength, uniqueValues, valueLength, randomSeed);
        return fragment;
    }

    /**
     * Sort a fragment
     *
     * @param fragment Fragment to be sorted
     * @return Sorted Fragment
     */
    public static Fragment sortPartition(Fragment fragment) {
        Fragment f = fragment.sort();
        return f;
    }

    /**
     * Filter a fragment values between a start and end range.
     *
     * @param fragment Fragment to filter
     * @param startValue Range start value
     * @param endValue Range end value
     * @return Fragment with a subset of values
     */
    public static Fragment filterTask(Fragment fragment, int startValue, int endValue) {
        Fragment f = fragment.filter(startValue, endValue);
        return f;
    }

    /**
     * Merge two fragments The contents of m2 are put into m1 and then m1
     * returned.
     *
     * @param m1 First fragment to merge
     * @param m2 Second fragment to merge
     * @return Fragment with the contents of m1 and m2
     */
    public static Fragment reduceTask(Fragment m1, Fragment m2) {
        m1.put(m2);
        return m1;
    }

}
