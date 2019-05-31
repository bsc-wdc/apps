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
package reverse;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.LinkedList;
import java.util.Queue;
import java.util.UUID;

public class Reverse {

	private static String tmpDir;

	public static void main(String args[]) throws Exception {

		boolean debug = Boolean.parseBoolean(args[0]);
		String inputDir = args[1];
		double chunks = Double.parseDouble(args[2]);
		String resultFile = args[3];
		tmpDir = args[4];

		long start = System.currentTimeMillis();

		if (debug) {
			System.out.println("Parameters: ");
			System.out.println("- Debug Enabled");
			System.out.println("- Input directory: " + inputDir);
			System.out.println("- Chunks: " + (int) chunks);
			System.out.println("- Result file: " + resultFile);
			System.out.println("- Temporary directory: " + tmpDir);
			System.out.println();
		}

		System.out.println("Processing tasks..." + (System.currentTimeMillis() - start));

		Queue<String> partials = new LinkedList<String>();

		File input = new File(inputDir);
		double size = input.listFiles().length;
		int chunk = (int) Math.ceil(size / chunks);
		int i = 0;

		System.out.println("Files read." + (System.currentTimeMillis() - start));

		while (i < size) {
			String partial = tmpDir + "/" + UUID.randomUUID().toString() + ".part";
			partials.add(partial);

			ReverseImpl.parseDir(inputDir, i, Math.min(i + chunk, (int) size), partial, debug);

			i += chunk;
		}

		while (partials.size() > 1) {
			String p1 = partials.poll();
			String p2 = partials.poll();

			ReverseImpl.mergeFiles(p1, p2);

			partials.add(p1);
		}

		System.out.println("Tasks processed." + (System.currentTimeMillis() - start));

		// write final result to file
		copyFile(partials.peek(), resultFile);
	}

	private static final void copyFile(String source, String target) throws IOException {
		int size = 8 * 1024;

		System.out.println(new File(source).length());

		FileInputStream fin = new FileInputStream(source);
		BufferedInputStream bin = new BufferedInputStream(fin, size);
		FileOutputStream fout = new FileOutputStream(target);
		BufferedOutputStream bout = new BufferedOutputStream(fout, size);
		int read;
		byte[] b = new byte[size];

		while ((read = bin.read(b)) >= 0)
			bout.write(b, 0, read);

		bin.close();
		bout.flush();
		bout.close();
	}
}
