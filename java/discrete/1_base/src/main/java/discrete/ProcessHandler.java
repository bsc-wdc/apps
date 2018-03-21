/*
 *  Copyright 2002-2015 Barcelona Supercomputing Center (www.bsc.es)
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
package discrete;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;


public class ProcessHandler {

    private Process proc;
    private OutputStream stdout;
    private OutputStream stderr;


    public ProcessHandler(Process proc, OutputStream stdout, OutputStream stderr) {
        this.proc = proc;
        this.stdout = stdout;
        this.stderr = stderr;
    }

    public int waitFor() {
        StreamReader out = new StreamReader(proc.getInputStream(), stdout);
        StreamReader err = new StreamReader(proc.getErrorStream(), stderr);

        out.start();
        err.start();

        int exit = -1;

        try {
            exit = proc.waitFor();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        out.interrupt();
        err.interrupt();

        return exit;
    }


    private class StreamReader extends Thread {

        private InputStream is;
        private OutputStream os;


        public StreamReader(InputStream is, OutputStream os) {
            this.is = is;
            this.os = os;
        }

        @Override
        public void run() {
            int read;
            byte[] b = new byte[1024];

            try {
                while (!Thread.currentThread().isInterrupted()) {
                    read = is.read(b);
                    if (read > 0 && os != null) {
                        os.write(b, 0, read);
                    }
                }
                is.close();
            } catch (IOException e) {
                // ignore
            }
        }
    }
    
}
