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

public class SortImpl {

    /**
     * Generate a Fragment object from the contents of a file
     *
     * @param filePath File with data to be sorted
     * @return Fragment which contains the data to be sorted
     */
    public static Fragment getFragment(String filePath) {
        Fragment fragment = new Fragment(filePath);
        return fragment;
    }

    /**
     * Filter a fragment values between a start and end range.
     *
     * @param fragment Fragment to filter
     * @param startValue Range start value
     * @param endValue Range end value
     * @return Fragment with a subset of values
     */
    public static Fragment filterTask(Fragment fragment, long startValue, long endValue) {
        Fragment f = fragment.filter(startValue, endValue);
        return f;
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

    /**
     * Save a fragment into a file
     *
     * @param fragment Fragment to save
     * @param filePath Destination file name
     * @return The number of elements written to the file
     */
    public static Integer saveFragment(Fragment fragment, String filePath) {
        return fragment.save(filePath);
    }

    /**
     * Merge two integers (i1 + i2)
     *
     * @param i1 First integer
     * @param i2 Second integer
     * @return The sum of i1 and i2
     */
    public static Integer reduceCount(Integer i1, Integer i2) {
        return i1 + i2;
    }

}
