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
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

import org.htmlparser.NodeFilter;
import org.htmlparser.Parser;
import org.htmlparser.filters.LinkRegexFilter;
import org.htmlparser.tags.LinkTag;
import org.htmlparser.util.EncodingChangeException;
import org.htmlparser.util.NodeList;
import org.htmlparser.util.ParserException;
import org.htmlparser.util.SimpleNodeIterator;

import com.ice.tar.TarEntry;
import com.ice.tar.TarInputStream;

public class ReverseImpl {

	private static final int BUFF_SIZE = 4096;

	public static void parse(String fileName, String html, String count) {

		System.out.println("Parsing file...");

		try {
			Parser parser = new Parser(html);
			NodeFilter filter = new LinkRegexFilter("http");
			NodeList nl = parser.parse(filter);
			SimpleNodeIterator it = nl.elements();
			Set<String> links = new HashSet<String>();

			// remove duplicates
			while (it.hasMoreNodes()) {
				LinkTag link = (LinkTag) it.nextNode();
				links.add(link.getLink());
			}

			FileOutputStream fos = new FileOutputStream(new File(count));

			System.out.println("Writing result to file...");

			// write result to file
			for (String l : links) {
				fos.write((l + " " + fileName + "\n").getBytes());
			}
			fos.close();

			System.out.println("Done!");

		} catch (ParserException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public static void parseDir(String directory, int begin, int end, String result, boolean debug) {

		long start = System.currentTimeMillis();

		File dir = new File(directory);
		int i = begin;
		File[] files = dir.listFiles();

		System.out.println("List time: " + (System.currentTimeMillis() - start));

		try {
			NodeFilter filter = new LinkRegexFilter("http");
			Map<String, String> indexing = new HashMap<String, String>();

			while (i < end) {

				File file = files[i];
				String fileName = file.getName();
				Parser parser;

				try {
					parser = new Parser(file.getAbsolutePath());

				} catch (ParserException e) {
					if (debug)
						System.out.println("Error parsing " + file.getAbsolutePath() + ". Omitting...");

					continue;
				}
				NodeList nl;

				try {
					if (debug)
						System.out.println("Parsing file " + (i - begin) + " of " + (end - begin));

					nl = parser.parse(filter);

				} catch (EncodingChangeException e) {
					if (debug)
						System.out.println("Encoding changed. Parsing again...");

					parser.reset();
					nl = parser.parse(filter);
				}
				SimpleNodeIterator it = nl.elements();
				Set<String> links = new HashSet<String>();

				// remove duplicates
				while (it.hasMoreNodes()) {
					LinkTag link = (LinkTag) it.nextNode();
					links.add(link.getLink());
				}
				for (String link : links) {
					if (indexing.containsKey(link)) {
						indexing.put(link, indexing.get(link) + " " + fileName);
					} else {
						indexing.put(link, fileName);
					}
				}
				i++;
			}
			mapToFile(indexing, result);

			System.out.println("ExecTime: " + (System.currentTimeMillis() - start));

		} catch (ParserException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	// public static void parseDir(String directory, int begin, int end, String
	// pack, boolean debug) {
	//
	// File dir = new File(directory);
	// int i = begin;
	// File[] files = dir.listFiles();
	//
	// try {
	// NodeFilter filter = new LinkRegexFilter("http");
	//
	// // File tarFile = new File("tmp.tar");
	// // TarOutputStream tar = new TarOutputStream(new
	// // FileOutputStream(tarFile));
	//
	// FileOutputStream zipfos = new FileOutputStream(pack);
	// ZipOutputStream zip = new ZipOutputStream(new
	// BufferedOutputStream(zipfos));
	//
	// while (i < end) {
	//
	// String fileName = files[i].getName();
	// Parser parser;
	//
	// try {
	// parser = new Parser(files[i].getAbsolutePath());
	//
	// } catch (ParserException e) {
	// if (debug)
	// System.out.println("Error parsing " + files[i].getAbsolutePath() +
	// ". Omitting...");
	//
	// continue;
	// }
	// NodeList nl;
	//
	// try {
	// if (debug)
	// System.out.println("Parsing file " + (i - begin) + " of " + (end -
	// begin));
	//
	// nl = parser.parse(filter);
	//
	// } catch (EncodingChangeException e) {
	// if (debug)
	// System.out.println("Encoding changed. Parsing again...");
	//
	// parser.reset();
	// nl = parser.parse(filter);
	// }
	// SimpleNodeIterator it = nl.elements();
	// Set<String> links = new HashSet<String>();
	//
	// // remove duplicates
	// while (it.hasMoreNodes()) {
	// LinkTag link = (LinkTag) it.nextNode();
	// links.add(link.getLink());
	// }
	//
	// String out = UUID.randomUUID().toString() + ".part";
	// File outFile = new File(out);
	//
	// FileOutputStream fos = new FileOutputStream(outFile);
	//
	// // write result to file
	// for (String l : links) {
	// fos.write((l + " " + fileName + "\n").getBytes());
	// }
	// fos.close();
	//
	// // if (outFile.length() > 0) {
	// // TarEntry entry = new TarEntry(out);
	// // entry.setSize(outFile.length());
	// // tar.putNextEntry(entry);
	// // copyStreams(new FileInputStream(outFile), tar);
	// // tar.closeEntry();
	// // }
	//
	// if (outFile.length() > 0) {
	// FileInputStream fi = new FileInputStream(outFile);
	// ZipEntry entry = new ZipEntry(outFile.getName());
	// zip.putNextEntry(entry);
	// BufferedInputStream buff = new BufferedInputStream(fi, BUFF_SIZE);
	// copyStreams(buff, zip);
	// buff.close();
	// }
	// outFile.delete();
	// i++;
	// }
	// zip.close();
	// // tar.close();
	//
	// // GZIPOutputStream gzip = new GZIPOutputStream(new
	// // FileOutputStream(pack));
	// //
	// // FileInputStream tarIn = new FileInputStream(tarFile);
	// //
	// // copyStreams(tarIn, gzip);
	// //
	// // gzip.close();
	// // tarIn.close();
	// // tarFile.delete();
	//
	// } catch (ParserException e) {
	// // TODO Auto-generated catch block
	// e.printStackTrace();
	// } catch (FileNotFoundException e) {
	// // TODO Auto-generated catch block
	// e.printStackTrace();
	// } catch (IOException e) {
	// // TODO Auto-generated catch block
	// e.printStackTrace();
	// }
	// }

	public static void mergeFiles(String f1, String f2) {

		System.out.println("Merging files...");

		try {
			Map<String, String> m1 = fileToMap(f1);
			Map<String, String> m2 = fileToMap(f2);

			for (String k2 : m2.keySet()) {
				if (m1.containsKey(k2)) {
					m1.put(k2, m1.get(k2) + " " + m2.get(k2));
				} else {
					m1.put(k2, m2.get(k2));
				}
			}

			System.out.println("Writing result to file...");

			mapToFile(m1, f1);

			System.out.println("Done!");

		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public static void mergePackage(String pack, String indexes) {

		try {
			FileInputStream fis = new FileInputStream(pack);
			ZipInputStream zin = new ZipInputStream(new BufferedInputStream(fis, BUFF_SIZE));

			// GZIPInputStream gzip1 = new GZIPInputStream(new
			// FileInputStream(pack));
			// File tarFile = new File("tmp.tar");
			// FileOutputStream tarOut = new FileOutputStream(tarFile);

			// copyStreams(gzip1, tarOut);
			//
			// gzip1.close();
			// tarOut.close();

			// TarInputStream tar = new TarInputStream(new
			// FileInputStream(tarFile));
			// TarEntry entry;
			Map<String, String> total = new HashMap<String, String>();

			ZipEntry entry;

			while ((entry = zin.getNextEntry()) != null) {

				String fileName = entry.getName();
				File partFile = new File(fileName);
				FileOutputStream fos = new FileOutputStream(partFile);
				BufferedOutputStream buff = new BufferedOutputStream(fos, BUFF_SIZE);
				copyStreams(zin, buff);
				buff.flush();
				buff.close();

				Map<String, String> part = fileToMap(fileName);

				for (String key : part.keySet()) {
					if (total.containsKey(key)) {
						total.put(key, total.get(key) + " " + part.get(key));
					} else {
						total.put(key, part.get(key));
					}
				}
				partFile.delete();
			}
			zin.close();

			mapToFile(total, indexes);
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	private static TarEntry nextEntry(TarInputStream tar) {
		while (true) {
			try {
				return tar.getNextEntry();

			} catch (Throwable e) {
				System.out.println("Skipping entry.");
			}
		}
	}

	private static Map<String, String> fileToMap(String f) throws IOException {

		Map<String, String> map = new HashMap<String, String>();

		BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(f)));

		String line;

		while ((line = br.readLine()) != null) {
			String[] tokens = line.split("\\s", 2);

			if (tokens.length == 2)
				map.put(tokens[0], tokens[1]);
		}
		br.close();

		return map;
	}

	private static void mapToFile(Map<String, String> m, String f) throws IOException {

		FileOutputStream fos = new FileOutputStream(new File(f));
		BufferedOutputStream bos = new BufferedOutputStream(fos);

		for (String k : m.keySet())
			bos.write((k + " " + m.get(k) + "\n").getBytes());

		bos.flush();
		bos.close();
	}

	private static void copyStreams(InputStream is, OutputStream os) throws IOException {
		byte[] b = new byte[BUFF_SIZE];
		int read;

		while ((read = is.read(b)) >= 0)
			os.write(b, 0, read);
	}

	// private static void compress(String fileName) throws IOException {
	// Process p = Runtime.getRuntime().exec("tar czvf " + fileName + " " +
	// TMP_DIR);
	//
	// try {
	// int out = p.waitFor();
	//
	// if (out != 0) {
	// copyStreams(p.getErrorStream(), System.out);
	// throw new RuntimeException("Unable to compress files.");
	// }
	//
	// } catch (InterruptedException e) {
	// // TODO Auto-generated catch block
	// e.printStackTrace();
	// }
	// }
	//
	// private static void uncompress(String file) throws IOException {
	//
	// Process p = Runtime.getRuntime().exec("tar xzvf " + file);
	//
	// try {
	// int out = p.waitFor();
	//
	// if (out != 0) {
	// copyStreams(p.getErrorStream(), System.out);
	// throw new RuntimeException("Unable to uncompress file: " + file);
	// }
	//
	// } catch (InterruptedException e) {
	// // TODO Auto-generated catch block
	// e.printStackTrace();
	// }
	// }

	public static final void main(String args[]) throws Exception {
		ReverseImpl.parseDir("test", 0, 3, "indexes", true);
	}
}
