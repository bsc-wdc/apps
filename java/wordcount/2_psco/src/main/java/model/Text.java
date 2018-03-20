package model;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;

import serialization.DataClayObject;


@SuppressWarnings("serial")
public class Text extends DataClayObject {

	private String title;
	private List<String> words1;
	// WordCollection words2;
	//private int maxNodeSize;
	private boolean debug;


	// Constructor required for COMPSs
	public Text() {
		System.out.println("[ Text ] Call to empty constructor (for COMPSs).");
	}

	// Constructor for the Wordcount application to get by alias
	public Text(int foo, String alias) {
		super(alias);
	}

	// Constructor for TextCollection to build new text objects
	public Text(String newTitle, final List<String> newWords, boolean doDebug) {
		title = newTitle;
		words1 = newWords;
		debug = doDebug;
		if (debug) {
			System.out.println("[ Text ] Call to real constructor to create a text object with " + words1.getClass().getName());
		}
	}

	// Constructor for TextCollection to build new text objects based on
	// WordCollection
	public Text(String newTitle, int maximumNodeSize) {
		title = newTitle;
		//maxNodeSize = maximumNodeSize;
	}

	public String getTitle() {
		return title;
	}

	public void addWords(String filePath) throws IOException {
		File file = new File(filePath);
		FileReader fr = new FileReader(file);
		BufferedReader br = new BufferedReader(fr);
		String line;
		int addedWords = 0;
		long totalSize = file.length();
		System.out.println("[ Text ] Parsing file " + file.getName() + " of size " + totalSize / 1024 / 1024 + " MB ...");
		long init = System.currentTimeMillis();
		while ((line = br.readLine()) != null) {
			String[] wordsLine = line.split(" ");
			for (String word : wordsLine) {
				words1.add(word);
				addedWords++;
			}
		}
		long end = System.currentTimeMillis();
		System.out.println("[ Text ] Added : " + addedWords + " words in " + (end - init) + " ms");

		br.close();
		fr.close();
	}

	public void wordCountFillingStats(TextStats textStatsToFill) {
		HashMap<String, Integer> result = new HashMap<String, Integer>();
		Iterator<String> it = words1.iterator();
		while (it.hasNext()) {
			String word = it.next();
			Integer curCount = result.get(word);
			if (curCount == null) {
				result.put(word, 1);
			} else {
				result.put(word, curCount + 1);
			}
		}
		textStatsToFill.setCurrentWordCount(result);
		if (debug) {
			System.out.println("[ Text ] ID of computed Text " + getID());
			System.out.println("[ Text ] ID of filled TextStats " + textStatsToFill.getID());
		}
	}

	public TextStats wordCountMakingPersistent() {
		HashMap<String, Integer> result = new HashMap<String, Integer>();
		Iterator<String> it = words1.iterator();
		while (it.hasNext()) {
			String word = it.next();
			Integer curCount = result.get(word);
			if (curCount == null) {
				result.put(word, 1);
			} else {
				result.put(word, curCount + 1);
			}
		}
		TextStats textStats = new TextStats(result);
		textStats.makePersistent("");
		if (debug) {
			System.out.println("[ Text ] ID of computed Text " + getID());
			System.out.println("[ Text ] ID of just created TextStats " + textStats.getID());
		}
		return textStats;
	}
}
