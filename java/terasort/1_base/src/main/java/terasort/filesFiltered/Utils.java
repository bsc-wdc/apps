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
package terasort.filesFiltered;

import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.IOException;

public class Utils {

    /**
     * Convert a byte array to long
     * @param b Byte array to convert
     * @return The long value of b
     */
    public static long byte2long(byte[] b) {
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

}
