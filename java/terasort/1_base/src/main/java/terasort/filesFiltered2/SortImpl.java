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

import java.util.ArrayList;

public class SortImpl {

    public static Fragment getFilteredFragment(String filePath, long startValue, long endValue) {
        Fragment fragment = new Fragment(filePath, startValue, endValue);
        return fragment;
    }

    public static Fragment sortPartition(Fragment fragment) {
        Fragment f = fragment.sort();
        return f;
    }

    public static Fragment reduceTask(Fragment m1, Fragment m2) {
        m1.put(m2);
        return m1;
    }

    public static Integer saveFragment(Fragment fragment, String filePath) {
        return fragment.fastSave(filePath);
    }

    public static Integer reduceCount(Integer i1, Integer i2) {
        return i1 + i2;
    }

    // Optimization
    public static void mixFragments(Fragment f1, Fragment f2, Range r, Fragment f1o, Fragment f2o) {
        // check that they have the same range
        long start = r.getStart();
        long end = r.getEnd();

        // divide the range in two halfs.
        long half = start + ((end - start) / 2);

        // move all elements lower than half to f1 and all higher to f2)
        ArrayList<Pair> lower = new ArrayList<Pair>();
        ArrayList<Pair> higher = new ArrayList<Pair>();

        // process first fragment
        ArrayList<Pair> f1l = f1.getFragment();
        int f1elems = f1l.size();
        for (int i = 0; i < f1elems; ++i) {
            Pair p = f1l.get(i);
            if (p.getKey() <= half) {
                lower.add(p);
            } else {
                higher.add(p);
            }
        }

        // process second fragment
        ArrayList<Pair> f2l = f2.getFragment();
        int f2elems = f2l.size();
        for (int i = 0; i < f2elems; ++i) {
            Pair p = f2l.get(i);
            if (p.getKey() <= half) {
                lower.add(p);
            } else {
                higher.add(p);
            }
        }

        Range lowRange = new Range(start, half);
        Range highRange = new Range(half, end);
        f1o.setRange(lowRange);
        f1o.setFragment(lower);
        f2o.setRange(highRange);
        f2o.setFragment(higher);

    }

}
