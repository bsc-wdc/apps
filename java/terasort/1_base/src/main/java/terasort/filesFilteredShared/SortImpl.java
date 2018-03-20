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
package terasort.filesFilteredShared;

import java.io.File;

public class SortImpl {

    public static Integer getFilteredFragment(String filePath, long startPos, long endPos, String[] bucketsPath, long bucketStep, int part) {
        int buckets = bucketsPath.length;
        int elems = -1;
        long bucketPos = 0;
        Fragment f = new Fragment(filePath, startPos, endPos);
        for (int i = 0; i < buckets; i++) {  // Create all buckets from the fragment contents.
            File b = new File(bucketsPath[i]);
            if (!b.exists()) {
                createFolder(b);
            }
            elems = f.filter(bucketPos, bucketPos + bucketStep).fastSave(bucketsPath[i] + "/" + part);
            part++;
            bucketPos += bucketStep;
        }
        return elems;
    }

    public static Integer reduceBuckets(Integer i1, Integer i2) {
        return i1 + i2;
    }

    public static Long sortBucket(String bucket, String output, Integer rangeElems) {
        File b = new File(bucket);
        // Read all files from a bucket
        File[] parts = b.listFiles();
        Fragment all = new Fragment();
        for (int i = 0; i < parts.length; i++) {
            all.put(new Fragment(parts[i].getAbsolutePath()));
        }
        // Sort the entire bucket
        all.sort();
        // Partition the sorted result in the same amoount of partitions as files where in the bucket.
        Fragment[] frags = all.partition(parts.length);
        for (int i = 0; i < parts.length; i++) {
            frags[i].fastSave(output + "_" + i + "_" + frags[i].getFragment().get(0).getKey());
        }
        // Remove the bucket (temporary files)
        deleteFolder(b);
        return all.size();
    }

    public static Long reduceSortedBuckets(Long i1, Long i2) {
        return i1 + i2;
    }

    public static void createFolder(File folder) {                            // not a task
        try {
            folder.mkdirs();
        } catch (SecurityException se) {
            System.out.println("[ERROR] Exception when creating the directory: " + folder.getAbsolutePath());
            System.out.println(se);
        }
    }

    public static void deleteFolder(File folder) {                            // not a task
        try {
            if (folder.isDirectory()) {
                String[] children = folder.list();
                for (int i = 0; i < children.length; i++) {
                    deleteFolder(new File(folder, children[i]));
                }
            }
            folder.delete(); // The directory is empty now and can be deleted.
        } catch (SecurityException se) {
            System.out.println("[ERROR] Exception when removing the directory: " + folder.getAbsolutePath());
            System.out.println(se);
        }
    }
}
