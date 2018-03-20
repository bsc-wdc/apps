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

import java.io.File;

public class SortImpl {

    public static Fragment readBlock(String filePath, long startPos, long endPos) {
        return new Fragment(filePath, startPos, endPos);
    }

    public static Fragment extractSortedBucket(Fragment f, long startBucket, long endBucket) {
        return f.filter(startBucket, endBucket);
    }

    public static Fragment mergeBuckets(Fragment f1, Fragment f2) {
        f1.put(f2); // puts and sorts
        return f1;
    }

    public static Long saveFragment(Fragment f, String filePath) {
        long elems = f.fastSave(filePath);
        return elems;
    }

    public static Long reduceSortedBucketsCount(Long i1, Long i2) {
        return i1 + i2;
    }

    // not a task
    public static void createFolder(File folder) {
        System.out.println("[LOG] Creating directory: " + folder.getAbsolutePath());
        boolean result = false;
        try {
            folder.mkdirs();
            result = true;
        } catch (SecurityException se) {
            System.out.println("[ERROR] Exception when creating the directory: " + folder.getAbsolutePath());
            System.out.println(se);
        }
        if (result) {
            System.out.println("[LOG] Folder succesfully created.");
        }
    }

    // not a task
    public static void deleteFolder(File folder) {
        //System.out.println("[LOG] Removing directory: " + folder.getAbsolutePath());
        boolean result = false;
        try {
            if (folder.isDirectory()) {
                String[] children = folder.list();
                for (int i = 0; i < children.length; i++) {
                    deleteFolder(new File(folder, children[i]));
                }
            }
            folder.delete(); // The directory is empty now and can be deleted.
            result = true;
        } catch (SecurityException se) {
            System.out.println("[ERROR] Exception when removing the directory: " + folder.getAbsolutePath());
            System.out.println(se);
        }
        //if(result) {    
        //    System.out.println("[LOG] Folder succesfully removed.");
        //}
    }

}
