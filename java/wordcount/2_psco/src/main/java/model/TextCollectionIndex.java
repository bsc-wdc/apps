package model;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import serialization.DataClayObject;


@SuppressWarnings("serial")
public class TextCollectionIndex extends DataClayObject {

	private ArrayList<TextCollection> textCollections;
	private int nextCollection;


	// Get by alias constructor for Wordcount app
	public TextCollectionIndex(final int foo, final String alias) {
		super(alias);
	}

	public TextCollectionIndex(ArrayList<TextCollection> newTextCollections) {
		textCollections = new ArrayList<TextCollection>();
		textCollections.addAll(newTextCollections);
		nextCollection = 0;
	}

	public ArrayList<String> getTextTitles() {
		ArrayList<String> result = new ArrayList<String>();
		for (TextCollection tc : textCollections) {
			result.addAll(tc.getTextTitles());
		}
		return result;
	}

	public int getSize() {
		int result = 0;
		for (TextCollection tc : textCollections) {
			result += tc.getSize();
		}
		return result;
	}

	public List<String> addTextsFromPath(final String filePath, List<String> emptyList) throws IOException {
		List<String> result;
		File f = new File(filePath);
		if (f.isDirectory()) {
			result = addTextsFromDir(filePath, emptyList);
		} else {
			result = new ArrayList<String>();
			String newTitle = addTextFromFile(filePath, emptyList);
			result.add(newTitle);
		}
		return result;
	}

	public String addTextFromFile(final String filePath, List<String> emptyList) throws IOException {
		if (nextCollection == textCollections.size()) {
			nextCollection = 0;
		}
		TextCollection tc = textCollections.get(nextCollection);
		nextCollection++;
		String textTitle = tc.addTextFromFile(filePath, emptyList);
		return textTitle;
	}

	public List<String> addTextsFromDir(final String dirPath, List<String> emptyList) throws IOException {
		File dir = new File(dirPath);
		List<String> result = new ArrayList<String>();
		for (File f : dir.listFiles()) {
			String addedText = addTextFromFile(f.getAbsolutePath(), emptyList);
			result.add(addedText);
		}
		return result;
	}
}
